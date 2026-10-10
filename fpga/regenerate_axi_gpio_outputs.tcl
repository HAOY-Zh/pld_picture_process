set project_file {D:/pld_picture_process/fpga/pynq_z2_window_level_axis/pynq_z2_window_level_axis.xpr}
open_project $project_file
set bd_file [get_files -quiet */design_1.bd]
open_bd_design $bd_file

reset_target all $bd_file
generate_target all $bd_file

set design_ips [get_ips]
foreach ip $design_ips {
    reset_target all $ip
}
generate_target all $design_ips

set wrapper_file [make_wrapper -files $bd_file -top -force]
if {[llength [get_files -quiet $wrapper_file]] == 0} {
    add_files -norecurse $wrapper_file
}
set_property top design_1_wrapper [get_filesets sources_1]
update_compile_order -fileset sources_1
save_bd_design
close_bd_design [get_bd_designs design_1]
close_project
exit
