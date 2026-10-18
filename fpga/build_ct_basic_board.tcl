if {$argc != 1} { error "Usage: build_ct_basic_board.tcl NEW_PREPARED_DIRECTORY" }
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
if {[string first "${root}/data/" $out] != 0 || [file exists $out/project]} { error "Need new workspace/data project" }
source [file join $out ps_settings.tcl]
create_project ct_basic [file join $out project] -part xc7z020clg400-1
set_property target_language Verilog [current_project]
foreach f {ct_basic_axis.v ct_window_pipeline.v threshold_range_axis.v threshold_range.v erosion_axis.v erosion_3x3.v dilation_axis.v dilation_3x3.v opening_axis.v closing_axis.v} {
    add_files -norecurse [file join $root fpga $f]
}
update_compile_order -fileset sources_1
# This installation's BD init sources helper scripts by relative name.
set previous_directory [pwd]
cd [file join $::env(XILINX_VIVADO) scripts ipintegrator]
create_bd_design design_1
cd $previous_directory
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
set_property -dict $ps_config $ps
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 1 CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 CONFIG.PCW_EN_CLK0_PORT 1 CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100] $ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} $ps
set core [create_bd_cell -type module -reference ct_basic_axis ct_basic_axis_0]
set dma [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0]
set_property -dict [list CONFIG.c_include_sg 0 CONFIG.c_sg_length_width 23 CONFIG.c_m_axis_mm2s_tdata_width 32 CONFIG.c_s_axis_s2mm_tdata_width 32 CONFIG.c_m_axi_mm2s_data_width 64 CONFIG.c_m_axi_s2mm_data_width 64 CONFIG.c_include_mm2s_dre 0 CONFIG.c_include_s2mm_dre 0] $dma
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_0]
set_property -dict [list CONFIG.C_GPIO_WIDTH 32 CONFIG.C_GPIO2_WIDTH 32 CONFIG.C_IS_DUAL 1 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_ALL_OUTPUTS_2 1] $gpio
set mode [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio:2.0 axi_gpio_1]
set_property -dict [list CONFIG.C_GPIO_WIDTH 3 CONFIG.C_ALL_OUTPUTS 1 CONFIG.C_DOUT_DEFAULT 0x00000007] $mode
set control [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 control_interconnect]
set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 3] $control
set memory [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 memory_interconnect]
set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $memory
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_GP0] [get_bd_intf_pins $control/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $control/M00_AXI] [get_bd_intf_pins $dma/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins $control/M01_AXI] [get_bd_intf_pins $gpio/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $control/M02_AXI] [get_bd_intf_pins $mode/S_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_MM2S] [get_bd_intf_pins $memory/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXI_S2MM] [get_bd_intf_pins $memory/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins $memory/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins $dma/M_AXIS_MM2S] [get_bd_intf_pins $core/s_axis]
connect_bd_intf_net [get_bd_intf_pins $core/m_axis] [get_bd_intf_pins $dma/S_AXIS_S2MM]
connect_bd_net [get_bd_pins $gpio/gpio_io_o] [get_bd_pins $core/window_params]
connect_bd_net [get_bd_pins $gpio/gpio2_io_o] [get_bd_pins $core/thresholds]
connect_bd_net [get_bd_pins $mode/gpio_io_o] [get_bd_pins $core/mode]
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] $reset
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_one]
set zero [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_zero]
set_property CONFIG.CONST_VAL 1 $one
set_property CONFIG.CONST_VAL 0 $zero
connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $ps/M_AXI_GP0_ACLK] [get_bd_pins $ps/S_AXI_HP0_ACLK] [get_bd_pins $dma/s_axi_lite_aclk] [get_bd_pins $dma/m_axi_mm2s_aclk] [get_bd_pins $dma/m_axi_s2mm_aclk] [get_bd_pins $control/aclk] [get_bd_pins $memory/aclk] [get_bd_pins $gpio/s_axi_aclk] [get_bd_pins $mode/s_axi_aclk] [get_bd_pins $core/clk] [get_bd_pins $reset/slowest_sync_clk]
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $reset/ext_reset_in]
connect_bd_net [get_bd_pins $one/dout] [get_bd_pins $reset/dcm_locked] [get_bd_pins $reset/aux_reset_in]
connect_bd_net [get_bd_pins $zero/dout] [get_bd_pins $reset/mb_debug_sys_rst]
connect_bd_net [get_bd_pins $reset/peripheral_aresetn] [get_bd_pins $dma/axi_resetn] [get_bd_pins $control/aresetn] [get_bd_pins $memory/aresetn] [get_bd_pins $gpio/s_axi_aresetn] [get_bd_pins $mode/s_axi_aresetn] [get_bd_pins $core/rst_n]
assign_bd_address
validate_bd_design
save_bd_design
generate_target all [get_files design_1.bd]
add_files -norecurse [make_wrapper -files [get_files design_1.bd] -top]
set_property top design_1_wrapper [current_fileset]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $out block_design.tcl]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Synthesis failed" }
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Implementation failed" }
open_run impl_1
report_utilization -file [file join $out utilization_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $out timing_routed.rpt]
report_drc -file [file join $out drc.rpt]
foreach delay {min max} {
    set path [get_timing_paths -delay_type $delay -max_paths 1]
    if {[llength $path] == 0 || [get_property SLACK $path] < 0} { error "Routed timing failed; do not deploy" }
}
file mkdir [file join $out deploy]
file copy [file join $out project ct_basic.runs impl_1 design_1_wrapper.bit] [file join $out deploy ct_basic.bit]
file copy [file join $out project ct_basic.gen sources_1 bd design_1 hw_handoff design_1.hwh] [file join $out deploy ct_basic.hwh]
puts "CT BASIC SHARED BIT BUILD PASS. Board functional test still required."
close_project
