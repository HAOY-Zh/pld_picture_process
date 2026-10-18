# Only immutable command configuration -> engine capture may take four cycles.
# lung_axil_control locks these registers during an explicit four-edge guard,
# then asserts VALID; the engine captures no earlier than the following edge.
# Live stream, weights, status/readback, state/control VALID and data AXI paths
# retain ordinary single-cycle timing. Do NOT apply a blanket clock exception.
set lung_cfg_launch [get_cells -hier -regexp {(^|.*/)control/(shape|geometry|input_address|output_address)_reg(\[[0-9]+\])?$}]
set lung_cfg_capture [filter [get_cells -hier -regexp {(^|.*/)engine/.*}] {IS_SEQUENTIAL}]
set_multicycle_path 4 -setup -from $lung_cfg_launch -to $lung_cfg_capture
set_multicycle_path 3 -hold -from $lung_cfg_launch -to $lung_cfg_capture
# Vivado can absorb/replicate configuration FFs into the count-multiply DSPs.
# These named products depend ONLY on immutable command geometry, not pixels.
# Give those relocated launch FFs three cycles (one less than raw config).
# Never include row_offset (live address generation) or any MAC/RQ DSP.
set lung_cfg_derived [get_cells -hier -regexp {(^|.*/)engine/(input_words0|output_words0|output_writer/command_words0|conv/conv/window/command_words0)$}]
set_multicycle_path 3 -setup -from $lung_cfg_derived -to $lung_cfg_capture
set_multicycle_path 2 -hold -from $lung_cfg_derived -to $lung_cfg_capture
