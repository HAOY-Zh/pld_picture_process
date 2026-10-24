if {$argc != 1} {error "Usage: build_overlay.tcl NEW_OUTPUT_DIRECTORY"}
set snapshot [file normalize [file dirname [info script]]]
set root [file normalize [file join $snapshot ../..]]
set workspace [file normalize [file join $snapshot ../../../..]]
set out [file normalize [lindex $argv 0]]
if {[string first "${workspace}/data/" $out] != 0 || [file exists $out]} {error "Use a fresh workspace/data output"}
file mkdir $out
set_param general.maxThreads 4
source [file join $root hardware ct_studio_20261018 basic_shared ps_settings.tcl]
create_project roi_statistics [file join $out project] -part xc7z020clg400-1
set_property target_language Verilog [current_project]
add_files -norecurse [list [file join $snapshot fpga pynq_z2_roi.v] [file join $snapshot fpga pynq_z2_roi_statistics.v]]
update_compile_order -fileset sources_1
set previous_directory [pwd]
cd [file join $::env(XILINX_VIVADO) scripts ipintegrator]
create_bd_design design_1
cd $previous_directory
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
set_property -dict $ps_config $ps
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 1 CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 CONFIG.PCW_EN_CLK0_PORT 1 CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100] $ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} $ps
set core [create_bd_cell -type module -reference pynq_z2_roi_statistics roi_statistics_0]
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list CONFIG.c_include_sg 0 CONFIG.c_sg_length_width 23 CONFIG.c_m_axis_mm2s_tdata_width 32 CONFIG.c_s_axis_s2mm_tdata_width 32 CONFIG.c_m_axi_mm2s_data_width 64 CONFIG.c_m_axi_s2mm_data_width 64 CONFIG.c_include_mm2s_dre 0 CONFIG.c_include_s2mm_dre 0] $dma
proc gpio {name direction1 direction2 width2} {
    set cell [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 $name]
    set config [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_IS_DUAL 1 CONFIG.C_GPIO2_WIDTH $width2]
    if {$direction1 eq "in"} {lappend config CONFIG.C_ALL_INPUTS 1} else {lappend config CONFIG.C_ALL_OUTPUTS 1}
    if {$direction2 eq "in"} {lappend config CONFIG.C_ALL_INPUTS_2 1} else {lappend config CONFIG.C_ALL_OUTPUTS_2 1}
    set_property -dict $config $cell
    return $cell
}
set geometry [gpio roi_geometry out out 32]
set bounds [gpio roi_bounds out out 32]
set hu_status [gpio roi_hu_status out in 32]
set count_identity [gpio roi_count_identity in in 32]
set sum [gpio roi_stats_sum in in 32]
set sumsq [gpio roi_stats_sumsq in in 32]
set meta [gpio roi_stats_meta in in 1]
set identity [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 roi_stats_identity]
set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_ALL_INPUTS 1] $identity
set control [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 control_interconnect]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 9] $control
set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 memory_interconnect]
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $memory
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_GP0] [get_bd_intf_pins $control/S00_AXI]
set slaves [list $dma $geometry $bounds $hu_status $count_identity $sum $sumsq $meta $identity]
set i 0
foreach slave $slaves {
    connect_bd_intf_net [get_bd_intf_pins $control/[format M%02d_AXI $i]] [get_bd_intf_pins $slave/[expr {$i==0 ? "S_AXI_LITE" : "S_AXI"}]]
    incr i
}
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $memory/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $memory/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $memory/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $core/s_axis]
connect_bd_intf_net [get_bd_intf_pins $core/m_axis] [get_bd_intf_pins $dma/S_AXIS_S2MM]
foreach {pin port} {
    roi_geometry/gpio_io_o geometry_in roi_geometry/gpio2_io_o modes_in
    roi_bounds/gpio_io_o bounds_x_in roi_bounds/gpio2_io_o bounds_y_in
    roi_hu_status/gpio_io_o thresholds_in roi_hu_status/gpio2_io_i status_out
    roi_count_identity/gpio_io_i count_out roi_count_identity/gpio2_io_i identity_out
    roi_stats_meta/gpio_io_i hu_minmax_out roi_stats_meta/gpio2_io_i statistics_valid_out
    roi_stats_identity/gpio_io_i statistics_identity_out
} {connect_bd_net [get_bd_pins $pin] [get_bd_pins $core/$port]}
foreach {name port lo hi dest} {
    sum_lo hu_sum_out 0 31 roi_stats_sum/gpio_io_i
    sum_hi hu_sum_out 32 63 roi_stats_sum/gpio2_io_i
    sumsq_lo hu_sumsq_out 0 31 roi_stats_sumsq/gpio_io_i
    sumsq_hi hu_sumsq_out 32 63 roi_stats_sumsq/gpio2_io_i
} {
    set slice [create_bd_cell -type ip -vlnv xilinx.com:ip:xlslice:1.0 $name]
    set_property -dict [list CONFIG.DIN_WIDTH 64 CONFIG.DIN_FROM $hi CONFIG.DIN_TO $lo CONFIG.DOUT_WIDTH 32] $slice
    connect_bd_net [get_bd_pins $core/$port] [get_bd_pins $slice/Din]
    connect_bd_net [get_bd_pins $slice/Dout] [get_bd_pins $dest]
}
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] $reset
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_one]
set zero [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_zero]
set_property CONFIG.CONST_VAL 1 $one
set_property CONFIG.CONST_VAL 0 $zero
set clocks [list $ps/M_AXI_GP0_ACLK $ps/S_AXI_HP0_ACLK $dma/s_axi_lite_aclk $dma/m_axi_mm2s_aclk $dma/m_axi_s2mm_aclk $control/aclk $memory/aclk $core/clk $reset/slowest_sync_clk]
foreach cell [lrange $slaves 1 end] {lappend clocks $cell/s_axi_aclk}
foreach pin $clocks {connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $pin]}
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $reset/ext_reset_in]
connect_bd_net [get_bd_pins $one/dout] [get_bd_pins $reset/dcm_locked] [get_bd_pins $reset/aux_reset_in]
connect_bd_net [get_bd_pins $zero/dout] [get_bd_pins $reset/mb_debug_sys_rst]
set resets [list $dma/axi_resetn $control/aresetn $memory/aresetn $core/rst_n]
foreach cell [lrange $slaves 1 end] {lappend resets $cell/s_axi_aresetn}
foreach pin $resets {connect_bd_net [get_bd_pins $reset/peripheral_aresetn] [get_bd_pins $pin]}
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]
add_files -norecurse [make_wrapper -files [get_files design_1.bd] -top]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $out block_design.tcl]
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -file [file join $out utilization_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $out timing_routed.rpt]
report_drc -file [file join $out drc.rpt]
set slacks {}
foreach delay {max min} {
    set path [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $path]==0 || [get_property SLACK $path]<0} {error "System routed timing failed; no deployable bit"}
    lappend slacks [get_property SLACK $path]
}
set f [open [file join $out timing_status.txt] w]
puts $f "WNS_NS=[lindex $slacks 0]\nWHS_NS=[lindex $slacks 1]\nCLOCK_MHZ=100\nSCOPE=FULL_ROI_PS_DMA_GPIO_OVERLAY"
close $f
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {error "Bitstream generation failed"}
file mkdir [file join $out deploy]
file copy [file join $out project roi_statistics.runs impl_1 design_1_wrapper.bit] [file join $out deploy ct_roi_statistics.bit]
set hwhs [concat [glob -nocomplain [file join $out project roi_statistics.srcs sources_1 bd design_1 hw_handoff design_1.hwh]] [glob -nocomplain [file join $out project roi_statistics.gen sources_1 bd design_1 hw_handoff design_1.hwh]]]
if {[llength $hwhs]!=1} {error "Expected one matching HWH"}
file copy [lindex $hwhs 0] [file join $out deploy ct_roi_statistics.hwh]
puts "ROI_STATISTICS_OVERLAY_READY: full PS/DMA/GPIO timing PASS; board acceptance pending"
close_project
