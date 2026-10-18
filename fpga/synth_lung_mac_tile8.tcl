# Isolated OOC synthesis of the cached 64-MAC core. No XPR/BD/board changes.
if {$argc != 1} { error "Usage: vivado -mode batch -source synth_lung_mac_tile8.tcl -tclargs NEW_OUTPUT_DIR" }
set task_root [file normalize [file join [file dirname [info script]] ..]]
set task_out [file normalize [lindex $argv 0]]
if {[string first "${task_root}/data/" $task_out] != 0} { error "Output must remain under workspace/data" }
if {[file exists $task_out]} { error "Output exists; choose a new directory" }
file mkdir $task_out
create_project -in_memory -part xc7z020clg400-1
read_verilog [file join $task_root fpga lung_requant.v]
read_verilog [file join $task_root fpga lung_mac_tile8.v]
read_verilog [file join $task_root fpga lung_mac_tile8_cached.v]
synth_design -top lung_mac_tile8_cached -part xc7z020clg400-1 -mode out_of_context
create_clock -name core_clk -period 10.000 [get_ports clk]
report_utilization -hierarchical -file [file join $task_out utilization_hier.rpt]
report_utilization -file [file join $task_out utilization.rpt]
report_timing_summary -delay_type max -max_paths 10 -file [file join $task_out timing_synth.rpt]
write_checkpoint [file join $task_out tile8_cached_synth.dcp]
puts "Cached tile8 OOC reports written; no routed timing, DMA, full NN or bitstream."
