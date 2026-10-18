# Non-project, isolated OOC synthesis. NEVER opens/edits the user's BD or XPR.
if {$argc != 1} { error "Usage: vivado -mode batch -source synth_lung_int8_dot.tcl -tclargs NEW_OUTPUT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Output must be a new task directory under workspace/data" }
if {[file exists $task_out]} { error "Output already exists; choose a new directory" }
file mkdir $task_out
create_project -in_memory -part xc7z020clg400-1
read_verilog [file join $task_root fpga lung_requant.v]
read_verilog [file join $task_root fpga lung_int8_dot.v]
synth_design -top lung_int8_dot -part xc7z020clg400-1 -mode out_of_context
create_clock -name core_clk -period 10.000 [get_ports clk]
report_utilization -file [file join $task_out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $task_out timing_synth.rpt]
write_checkpoint [file join $task_out lung_int8_dot_synth.dcp]
puts "Completed isolated OOC synthesis; post-synth estimates only, no routed timing/bitstream."
