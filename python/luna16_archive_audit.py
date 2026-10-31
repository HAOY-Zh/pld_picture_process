"""Read-only audit of a repacked LUNA16 ZIP; prepare one explicitly labelled case.

The source archive/annotations are never changed. Pixel and ZIP checks establish
internal integrity, not identity with the original challenge download or licensing.
No training, candidate generation, segmentation or board access is performed.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
from time import perf_counter
from zipfile import ZipFile

import numpy as np


UID_RE = re.compile(r"[0-9]+(?:\.[0-9]+)+")
MAX_CASE_BYTES = 1024 ** 3


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def parse_header(raw, uid):
    values = {}
    for line in raw.decode("ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in values:
            raise ValueError("Duplicate MetaImage field")
        values[key.strip()] = value.strip()
    required = {"NDims", "BinaryData", "ElementType", "ElementDataFile", "DimSize",
                "ElementSpacing", "TransformMatrix", "Offset"}
    if not required <= set(values):
        raise ValueError("Incomplete MetaImage header")
    if (values["NDims"] != "3" or values["BinaryData"].lower() != "true" or
            values.get("CompressedData", "False").lower() != "false" or
            values["ElementType"] != "MET_SHORT" or
            values.get("HeaderSize", "0") != "0" or
            values.get("ElementNumberOfChannels", "1") != "1"):
        raise ValueError("This audit supports uncompressed, single-channel, signed int16 3-D MetaImage")
    if values["ElementDataFile"] != uid + ".raw":
        raise ValueError("Unexpected pixel filename; external paths are not followed")
    shape_xyz = np.array([int(x) for x in values["DimSize"].split()])
    spacing_xyz = np.array([float(x) for x in values["ElementSpacing"].split()])
    origin = np.array([float(x) for x in values["Offset"].split()])
    direction = np.array([float(x) for x in values["TransformMatrix"].split()]).reshape(3, 3)
    if (shape_xyz.shape != (3,) or spacing_xyz.shape != (3,) or origin.shape != (3,) or
            (shape_xyz < 1).any() or (shape_xyz > 2048).any() or
            not np.isfinite(spacing_xyz).all() or (spacing_xyz <= 0).any() or
            not np.isfinite(origin).all() or not np.isfinite(direction).all() or
            not np.allclose(direction.T @ direction, np.eye(3), atol=1e-6, rtol=0)):
        raise ValueError("Invalid shape, spacing, origin or direction")
    byte_count = int(np.prod(shape_xyz)) * 2
    if not 0 < byte_count <= MAX_CASE_BYTES:
        raise ValueError("Case exceeds audit size limit")
    basis = (direction * spacing_xyz)[..., ::-1]
    endian = values.get("BinaryDataByteOrderMSB", values.get("ElementByteOrderMSB", "False")).lower()
    if endian not in ("true", "false"):
        raise ValueError("Invalid byte order")
    return dict(shape_zyx=shape_xyz[::-1].tolist(), spacing_zyx_mm=spacing_xyz[::-1].tolist(),
                origin_lps_mm=origin.tolist(), direction_xyz=direction.tolist(),
                basis_lps_zyx=basis.tolist(), dtype=">i2" if endian == "true" else "<i2",
                raw_bytes=byte_count, header=values)


def audit_case(outer, entry, annotations):
    name = PurePosixPath(entry.filename)
    uid = name.name.removesuffix(".mhd.zip")
    if not UID_RE.fullmatch(uid) or name.name != entry.filename or not entry.filename.endswith(".mhd.zip"):
        raise ValueError("Unexpected outer archive member")
    if not 0 < entry.file_size <= MAX_CASE_BYTES:
        raise ValueError("Nested archive exceeds size limit")
    nested_bytes = outer.read(entry)  # Full outer member read also verifies its CRC.
    with ZipFile(BytesIO(nested_bytes)) as inner:
        members = [x for x in inner.infolist() if not x.is_dir()]
        expected = {uid + ".mhd", uid + ".raw"}
        if len(members) != 2 or {x.filename for x in members} != expected:
            raise ValueError("Expected exactly one matching .mhd/.raw pair")
        if inner.getinfo(uid + ".mhd").file_size > 65536:
            raise ValueError("Oversized MetaImage header")
        header = inner.read(uid + ".mhd")  # Inner header CRC is checked here.
        result = parse_header(header, uid)
        raw_info = inner.getinfo(uid + ".raw")
        if raw_info.file_size != result["raw_bytes"]:
            raise ValueError("Pixel bytes do not match dimensions/type")
        digest = hashlib.sha256()
        low, high, count = 32767, -32768, 0
        with inner.open(raw_info) as stream:
            while block := stream.read(4 * 1024 ** 2):
                digest.update(block)
                pixels = np.frombuffer(block, dtype=result["dtype"])
                low = min(low, int(pixels.min()))
                high = max(high, int(pixels.max()))
                count += len(block)
        if count != result["raw_bytes"]:
            raise ValueError("Truncated pixel payload")
    nodules = []
    basis = np.array(result["basis_lps_zyx"])
    origin = np.array(result["origin_lps_mm"])
    for row in annotations.get(uid, []):
        world = np.array([float(row[k]) for k in ("coordX", "coordY", "coordZ")])
        diameter = float(row["diameter_mm"])
        if not np.isfinite(world).all() or not np.isfinite(diameter) or diameter <= 0:
            raise ValueError("Invalid reference annotation")
        index = np.linalg.solve(basis, world - origin)
        nodules.append(dict(center_lps_xyz_mm=world.tolist(), center_index_zyx=index.tolist(),
                            diameter_mm=diameter, center_in_bounds=bool(
                                ((index >= -.5) & (index <= np.array(result["shape_zyx"]) - .5)).all())))
    direction = np.array(result["direction_xyz"])
    axis_aligned = bool(np.allclose(direction, np.diag(np.diag(direction)), atol=1e-6, rtol=0))
    result.update(series_uid=uid, source_member=entry.filename, outer_member_crc32=f"{entry.CRC:08x}",
                  outer_and_inner_crc="PASS", raw_sha256=digest.hexdigest(), hu_min=low, hu_max=high,
                  nodules=nodules, reference_nodule_count=len(nodules),
                  existing_viewer_axis_aligned=axis_aligned,
                  existing_viewer_size_ok=bool(np.prod(result["shape_zyx"]) <= 128 * 1024 ** 2 and
                      result["shape_zyx"][1] >= 16 and result["shape_zyx"][2] >= 16))
    if any(not n["center_in_bounds"] for n in nodules):
        raise ValueError("Reference center outside native image grid")
    return result


def prepare_case(source, case, folder):
    import SimpleITK as sitk
    from lung_live_data import read_ct

    folder.mkdir()
    uid = case["series_uid"]
    with ZipFile(source) as outer, ZipFile(BytesIO(outer.read(case["source_member"]))) as inner:
        for suffix in (".mhd", ".raw"):
            data = inner.read(uid + suffix)
            if suffix == ".raw" and hashlib.sha256(data).hexdigest() != case["raw_sha256"]:
                raise ValueError("Pixels changed since audit")
            (folder / (uid + suffix)).write_bytes(data)
    image = sitk.ReadImage(str(folder / (uid + ".mhd")))
    hu = sitk.GetArrayFromImage(image)
    if hu.dtype != np.int16 or list(hu.shape) != case["shape_zyx"]:
        raise ValueError("SimpleITK shape/type disagrees with header audit")
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    basis = (direction * spacing)[:, ::-1]
    if (not np.allclose(basis, case["basis_lps_zyx"], atol=1e-9, rtol=0) or
            not np.allclose(image.GetOrigin(), case["origin_lps_mm"], atol=1e-9, rtol=0)):
        raise ValueError("SimpleITK physical geometry disagrees with audit")
    if hashlib.sha256(hu.astype(case["dtype"], copy=False).tobytes()).hexdigest() != case["raw_sha256"]:
        raise ValueError("SimpleITK pixels disagree with audited original payload")
    orientation = np.concatenate((direction[:, 0], direction[:, 1]))
    if not np.allclose(np.cross(direction[:, 0], direction[:, 1]), direction[:, 2], atol=1e-9, rtol=0):
        raise ValueError("Reflected geometry needs explicit supported conversion")
    npz = folder / "ct_hu.npz"
    np.savez_compressed(npz, hu=hu, spacing_zyx_mm=spacing[::-1],
                        origin_lps_mm=np.array(image.GetOrigin()), orientation=orientation)
    restored, geom, details = read_ct(npz, confirm_hu=True)
    if (not np.array_equal(restored, hu) or
            not np.allclose(geom["basis_lps_zyx"], basis, atol=1e-9, rtol=0) or
            not np.allclose(geom["origin_lps_mm"], case["origin_lps_mm"], atol=1e-9, rtol=0)):
        raise ValueError("Existing web reader NPZ roundtrip failed")
    del restored
    reference_file = folder / "reference_nodules.json"
    save_json(reference_file, dict(series_uid=uid, annotations_role="REFERENCE_ONLY_NOT_MODEL_INPUT",
                                  nodules=case["nodules"], voxel_segmentation_ground_truth=False))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Ellipse
    plots = []
    for i, nodule in enumerate(case["nodules"]):
        z, y, x = nodule["center_index_zyx"]
        iz = int(np.rint(z))
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(hu[iz], cmap="gray", vmin=-1350, vmax=150, origin="upper")
        diameter = nodule["diameter_mm"]
        ax.add_patch(Ellipse((x, y), diameter / spacing[0], diameter / spacing[1],
                             fill=False, edgecolor="lime", linewidth=1.5))
        ax.plot(x, y, "+", color="yellow", markersize=9)
        ax.set_title(f"Reference center/diameter only; axial z={iz}\nNot a prediction or voxel segmentation")
        ax.set_axis_off()
        plot = folder / f"reference_location_{i+1:02d}_z{iz:03d}.png"
        fig.savefig(plot, dpi=130, bbox_inches="tight")
        plt.close(fig)
        plots.append(str(plot.resolve()))
    return dict(series_uid=uid, npz=str(npz.resolve()), source_mhd=str((folder / (uid + ".mhd")).resolve()),
                reference_json=str(reference_file.resolve()), reference_plots=plots,
                roundtrip="PASS: SimpleITK pixels/geometry and existing read_ct",
                web_hu_sha256=details["hu_sha256"], annotations_in_npz=False)


def run(args):
    source = args.archive.resolve(strict=True)
    annotation_path = args.annotations.resolve(strict=True)
    out = args.outdir.resolve()
    if out.exists():
        raise ValueError("Use a new output directory; previous audits must not be overwritten")
    out.mkdir(parents=True)
    annotations = {}
    with annotation_path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            annotations.setdefault(row["seriesuid"], []).append(row)
    start = perf_counter()
    source_stat = source.stat()
    report = dict(status="RUNNING", source_archive=str(source), source_bytes=source_stat.st_size,
                  source_mtime_ns=source_stat.st_mtime_ns, source_changed=False,
                  annotations=str(annotation_path), annotations_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                  official_pixel_identity_verified=False, original_license_review="separate_from_integrity_audit",
                  annotation_scope="center_and_diameter_only; no voxel nodule ground truth",
                  hu_provenance="LUNA16 mirror; values preserved, not inferred solely from intensity range",
                  cases=[], errors=[])
    with ZipFile(source) as outer:
        entries = [x for x in outer.infolist() if not x.is_dir()]
        report["archive_case_members"] = len(entries)
        if len({x.filename for x in entries}) != len(entries):
            raise ValueError("Duplicate archive members")
        for i, entry in enumerate(entries):
            try:
                report["cases"].append(audit_case(outer, entry, annotations))
            except Exception as error:
                report["errors"].append(dict(member=entry.filename, error=f"{type(error).__name__}: {error}"))
            if (i + 1) % 5 == 0 or i + 1 == len(entries):
                print(f"Audited {i+1}/{len(entries)}; errors={len(report['errors'])}; {perf_counter()-start:.1f}s", flush=True)
                save_json(out / "audit.json", report)
    ids = [c["series_uid"] for c in report["cases"]]
    if len(ids) != len(set(ids)):
        report["errors"].append(dict(error="Duplicate series UIDs"))
    report["summary"] = dict(valid_cases=len(ids), reference_nodules=sum(c["reference_nodule_count"] for c in report["cases"]),
                           cases_with_reference_nodules=sum(c["reference_nodule_count"] > 0 for c in report["cases"]),
                           cases_without_reference_nodules=sum(c["reference_nodule_count"] == 0 for c in report["cases"]),
                           raw_bytes=sum(c["raw_bytes"] for c in report["cases"]),
                           hu_min=min((c["hu_min"] for c in report["cases"]), default=None),
                           hu_max=max((c["hu_max"] for c in report["cases"]), default=None),
                           viewer_compatible_cases=sum(c["existing_viewer_axis_aligned"] and c["existing_viewer_size_ok"] for c in report["cases"]))
    if not report["errors"] and args.prepare_first:
        case = next(c for c in report["cases"] if c["nodules"] and c["existing_viewer_axis_aligned"] and c["existing_viewer_size_ok"])
        try:
            report["prepared_case"] = prepare_case(source, case, out / "first_case")
        except Exception as error:
            report["errors"].append(dict(stage="prepare_first", error=f"{type(error).__name__}: {error}"))
    final_stat = source.stat()
    report["source_changed"] = (final_stat.st_size, final_stat.st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns)
    if report["source_changed"]:
        report["errors"].append(dict(error="Source archive changed during audit"))
    report["status"] = "PASS" if not report["errors"] else "FAIL"
    report["elapsed_seconds"] = perf_counter() - start
    save_json(out / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cases",)}, indent=2), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--prepare-first", action="store_true")
    run(parser.parse_args())
