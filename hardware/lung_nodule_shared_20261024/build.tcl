# Separate PYNQ-Z2 dual-engine layer overlay; keep the proven single-block project unchanged.
# First argument: freshly prepared folder containing ps_settings.tcl.
# Optional second argument: "build" runs synthesis/implementation/bit generation.
if {$argc < 1 || $argc > 2} { error "Usage: build_lung_network_board.tcl PREPARED_DIR ?build?" }
set snapshot [file normalize [file dirname [info script]]]
set task_root [file normalize [file join $snapshot ../..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Output outside workspace/data" }
if {[file exists [file join $task_out project]]} { error "Need a new prepared directory" }
source [file join $snapshot ps_settings.tcl]
create_project lung_network [file join $task_out project] -part xc7z020clg400-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set rtl_names {lung_requant.v lung_mac_tile8.v lung_mac_tile8_reuse.v lung_feature_window.v lung_conv8_window.v lung_feature_cache.v lung_axi_read64.v lung_conv8_cached_axi.v lung_axi_write64.v lung_output_writer.v lung_conv8_memory_axi.v lung_axil_control.v lung_conv8_axil.v lung_tensor_burst_engine.v lung_axi_burst_write64.v lung_tensor_control.v lung_tensor_axil.v}
foreach f $rtl_names { add_files -norecurse [file join $snapshot rtl $f] }
set_property top lung_conv8_axil [current_fileset]
update_compile_order -fileset sources_1
set repo [file join $task_out ip_repo]
foreach top_name {lung_conv8_axil lung_tensor_axil} {
    set_property top $top_name [current_fileset]
ipx::package_project -root_dir [file join $repo $top_name] -vendor teaching.local -library user -taxonomy /UserIP -import_files
set core [ipx::current_core]
set_property name $top_name $core
set_property version 1.0 $core
set_property display_name "Lung layer engine $top_name" $core
proc bus_parameter {bus name value} {
    set p [ipx::get_bus_parameters $name -of_objects $bus]
    if {[llength $p] == 0} { set p [ipx::add_bus_parameter $name $bus] }
    set_property value $value $p
}
proc make_bus {core name mode busdef abstraction maps} {
    set bus [ipx::get_bus_interfaces $name -of_objects $core]
    if {[llength $bus] != 0} { ipx::remove_bus_interface $name $core }
    set bus [ipx::add_bus_interface $name $core]
    set_property interface_mode $mode $bus
    set_property bus_type_vlnv $busdef $bus
    set_property abstraction_type_vlnv $abstraction $bus
    foreach {logical physical} $maps {
        set p [ipx::add_port_map $logical $bus]
        set_property physical_name $physical $p
    }
    return $bus
}
set smap {};foreach name {AWADDR AWPROT AWVALID AWREADY WDATA WSTRB WVALID WREADY BRESP BVALID BREADY ARADDR ARPROT ARVALID ARREADY RDATA RRESP RVALID RREADY} { lappend smap $name s_axi_[string tolower $name] }
set sbus [make_bus $core s_axi slave xilinx.com:interface:aximm:1.0 xilinx.com:interface:aximm_rtl:1.0 $smap]
bus_parameter $sbus PROTOCOL AXI4LITE;bus_parameter $sbus DATA_WIDTH 32;bus_parameter $sbus ADDR_WIDTH 10
set mmap [ipx::get_memory_maps s_axi -of_objects $core]
if {[llength $mmap] == 0} { set mmap [ipx::add_memory_map s_axi $core] }
set block [ipx::get_address_blocks Reg -of_objects $mmap]
if {[llength $block] == 0} { set block [ipx::add_address_block Reg $mmap] }
set_property base_address 0 $block;set_property range 1024 $block;set_property width 32 $block;set_property usage register $block
set_property slave_memory_map_ref s_axi $sbus
set mmaplist {};foreach name {ARID ARADDR ARLEN ARSIZE ARBURST ARLOCK ARCACHE ARPROT ARVALID ARREADY RID RDATA RRESP RLAST RVALID RREADY AWID AWADDR AWLEN AWSIZE AWBURST AWLOCK AWCACHE AWPROT AWVALID AWREADY WDATA WSTRB WLAST WVALID WREADY BID BRESP BVALID BREADY} { lappend mmaplist $name m_axi_[string tolower $name] }
set mbus [make_bus $core m_axi master xilinx.com:interface:aximm:1.0 xilinx.com:interface:aximm_rtl:1.0 $mmaplist]
foreach {name value} {PROTOCOL AXI4 DATA_WIDTH 64 ADDR_WIDTH 32 ID_WIDTH 4 NUM_READ_OUTSTANDING 1 NUM_WRITE_OUTSTANDING 1 MAX_BURST_LENGTH 16 SUPPORTS_NARROW_BURST 0} { bus_parameter $mbus $name $value }
set space [ipx::get_address_spaces m_axi -of_objects $core]
if {[llength $space] == 0} { set space [ipx::add_address_space m_axi $core] }
set_property range 4294967296 $space;set_property width 32 $space;set_property master_address_space_ref m_axi $mbus
set clockbus [make_bus $core clk slave xilinx.com:signal:clock:1.0 xilinx.com:signal:clock_rtl:1.0 {CLK clk}]
bus_parameter $clockbus ASSOCIATED_BUSIF s_axi:m_axi;bus_parameter $clockbus ASSOCIATED_RESET rst_n;bus_parameter $clockbus FREQ_HZ 100000000
set resetbus [make_bus $core rst_n slave xilinx.com:signal:reset:1.0 xilinx.com:signal:reset_rtl:1.0 {RST rst_n}]
bus_parameter $resetbus POLARITY ACTIVE_LOW
ipx::create_xgui_files $core
ipx::update_checksums $core
ipx::check_integrity $core
ipx::save_core $core
ipx::unload_core $core

}
set_property ip_repo_paths $repo [current_project]
update_ip_catalog
create_bd_design design_1
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0]
set_property -dict $ps_config $ps
set_property -dict [list CONFIG.PCW_USE_M_AXI_GP0 1 CONFIG.PCW_USE_S_AXI_HP0 1 CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 CONFIG.PCW_EN_CLK0_PORT 1 CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100] $ps
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} $ps
set accel [create_bd_cell -type ip -vlnv teaching.local:user:lung_conv8_axil:1.0 lung_conv8_axil_0]
set tensor [create_bd_cell -type ip -vlnv teaching.local:user:lung_tensor_axil:1.0 lung_tensor_axil_0]
foreach name {control_interconnect memory_interconnect} {
    set sc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 $name]
    if {$name eq "control_interconnect"} {
        set_property -dict [list CONFIG.NUM_SI 1 CONFIG.NUM_MI 2] $sc
    } else {
        set_property -dict [list CONFIG.NUM_SI 2 CONFIG.NUM_MI 1] $sc
    }
}
connect_bd_intf_net [get_bd_intf_pins $ps/M_AXI_GP0] [get_bd_intf_pins control_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_interconnect/M00_AXI] [get_bd_intf_pins $accel/s_axi]
connect_bd_intf_net [get_bd_intf_pins $accel/m_axi] [get_bd_intf_pins memory_interconnect/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins control_interconnect/M01_AXI] [get_bd_intf_pins $tensor/s_axi]
connect_bd_intf_net [get_bd_intf_pins $tensor/m_axi] [get_bd_intf_pins memory_interconnect/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins memory_interconnect/M00_AXI] [get_bd_intf_pins $ps/S_AXI_HP0]
set reset [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0]
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] $reset
set one [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_one];set_property CONFIG.CONST_VAL 1 $one
set zero [create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_zero];set_property CONFIG.CONST_VAL 0 $zero
connect_bd_net [get_bd_pins $ps/FCLK_CLK0] [get_bd_pins $ps/M_AXI_GP0_ACLK] [get_bd_pins $ps/S_AXI_HP0_ACLK] [get_bd_pins $accel/clk] [get_bd_pins $tensor/clk] [get_bd_pins control_interconnect/aclk] [get_bd_pins memory_interconnect/aclk] [get_bd_pins $reset/slowest_sync_clk]
connect_bd_net [get_bd_pins $ps/FCLK_RESET0_N] [get_bd_pins $reset/ext_reset_in]
# Both external and auxiliary reset inputs are ACTIVE LOW. Unused auxiliary
# reset must be HIGH; grounding it holds every AXI endpoint in reset forever.
connect_bd_net [get_bd_pins $one/dout] [get_bd_pins $reset/dcm_locked] [get_bd_pins $reset/aux_reset_in]
connect_bd_net [get_bd_pins $zero/dout] [get_bd_pins $reset/mb_debug_sys_rst]
connect_bd_net [get_bd_pins $reset/peripheral_aresetn] [get_bd_pins $accel/rst_n] [get_bd_pins $tensor/rst_n] [get_bd_pins control_interconnect/aresetn] [get_bd_pins memory_interconnect/aresetn]
assign_bd_address -offset 0x43c00000 -range 1K -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $accel/s_axi/Reg]
assign_bd_address -offset 0 -range 512M -target_address_space [get_bd_addr_spaces $accel/m_axi] [get_bd_addr_segs $ps/S_AXI_HP0/HP0_DDR_LOWOCM]
assign_bd_address -offset 0x43c10000 -range 1K -target_address_space [get_bd_addr_spaces $ps/Data] [get_bd_addr_segs $tensor/s_axi/Reg]
assign_bd_address -offset 0 -range 512M -target_address_space [get_bd_addr_spaces $tensor/m_axi] [get_bd_addr_segs $ps/S_AXI_HP0/HP0_DDR_LOWOCM]
validate_bd_design
source [file join $snapshot validate_lung_window_step.tcl]
if {[get_property CONFIG.C_AUX_RESET_HIGH $reset] != 0 || [get_property CONFIG.C_EXT_RESET_HIGH $reset] != 0} { error "Unexpected reset polarity" }
set aux_net [get_bd_nets -of_objects [get_bd_pins $reset/aux_reset_in]]
if {$aux_net ne [get_bd_nets -of_objects [get_bd_pins $one/dout]]} { error "Auxiliary reset not tied inactive" }
save_bd_design
generate_target all [get_files design_1.bd]
set wrapper [make_wrapper -files [get_files design_1.bd] -top]
add_files -norecurse $wrapper
set_property top design_1_wrapper [current_fileset]
add_files -fileset constrs_1 -norecurse [file join $snapshot lung_axil_config_timing.xdc]
set_property USED_IN_SYNTHESIS false [get_files lung_axil_config_timing.xdc]
set_property PROCESSING_ORDER LATE [get_files lung_axil_config_timing.xdc]
update_compile_order -fileset sources_1
write_bd_tcl -force [file join $task_out block_design.tcl]
if {$argc == 2 && [lindex $argv 1] eq "build"} {
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { error "Synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {![string match "*route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Implementation did not finish route_design"
}
open_run impl_1
report_utilization -file [file join $task_out utilization_routed.rpt]
report_timing_summary -delay_type min_max -max_paths 10 -file [file join $task_out timing_routed.rpt]
report_drc -file [file join $task_out drc.rpt]
report_exceptions -file [file join $task_out configuration_exceptions.rpt]
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup] == 0 || [get_property SLACK $setup] < 0 ||
    [llength $hold] == 0 || [get_property SLACK $hold] < 0} {
    error "Routed setup/hold failed; no bitstream generation"
}
foreach v [get_drc_violations -quiet] {
    if {[get_property SEVERITY $v] in {Error {Critical Warning}}} { error "DRC failed: $v" }
}
close_design
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Bitstream generation failed" }
file mkdir [file join $task_out deploy]
file copy [file join $task_out project lung_network.runs impl_1 design_1_wrapper.bit] [file join $task_out deploy lung_network.bit]
file copy [file join $task_out project lung_network.gen sources_1 bd design_1 hw_handoff design_1.hwh] [file join $task_out deploy lung_network.hwh]
puts "Resumed route and timing-gated bit generation complete; independent package audit and board test still required."
}
close_project
