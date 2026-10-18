# Actual proc_sys_reset vendor IP, not a replacement behavioural model.
if {$argc != 1} { error "Usage: simulate_lung_board_reset.tcl NEW_OUTPUT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0 || [file exists [file join $task_out project]]} { error "Need a fresh directory under data" }
create_project reset_check [file join $task_out project] -part xc7z020clg400-1
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
create_ip -name proc_sys_reset -vendor xilinx.com -library ip -version 5.0 -module_name lung_reset_check
set_property -dict [list CONFIG.C_EXT_RESET_HIGH 0 CONFIG.C_AUX_RESET_HIGH 0] [get_ips lung_reset_check]
generate_target all [get_ips lung_reset_check]
add_files -norecurse [file join $task_root fpga lung_axil_control.v]
add_files -fileset sim_1 -norecurse [file join $task_root fpga tb_lung_board_reset.sv]
set_property top tb_lung_board_reset [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim
set log [file join $task_out project reset_check.sim sim_1 behav xsim simulate.log]
set handle [open $log r];set transcript [read $handle];close $handle
if {[string first "Board vendor reset RTL: PASS" $transcript] < 0 || [regexp {Fatal:|ERROR:} $transcript]} { error "Actual vendor reset check failed; inspect simulate.log" }
puts "Vendor reset startup and AXI-Lite ID proof completed."
close_project
