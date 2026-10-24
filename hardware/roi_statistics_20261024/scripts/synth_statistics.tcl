set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir [file join $root results fpga_statistics]
create_project -in_memory -part xc7z020clg400-1
read_verilog [file join $root fpga pynq_z2_roi.v]
read_verilog [file join $root fpga pynq_z2_roi_statistics.v]
synth_design -top pynq_z2_roi_statistics -mode out_of_context -part xc7z020clg400-1
create_clock -name clk -period 10.000 [get_ports clk]
report_utilization -file [file join $root results fpga_statistics utilization_synth.rpt]
# Standalone routed timing only; no board top/DDR/PS or pin constraints are added.
opt_design
place_design
route_design
report_timing_summary -file [file join $root results fpga_statistics timing_routed.rpt]
report_utilization -file [file join $root results fpga_statistics utilization_routed.rpt]
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
set wns [get_property SLACK $setup_paths]
set whs [get_property SLACK $hold_paths]
set f [open [file join $root results fpga_statistics timing_status.txt] w]
puts $f "WNS_NS=$wns\nWHS_NS=$whs\nCLOCK_MHZ=100\nSCOPE=OUT_OF_CONTEXT_ONLY"
close $f
if {$wns < 0 || $whs < 0} {error "Statistics standalone timing did not close"}
puts "ROI_STATISTICS_OOC_PASS"
