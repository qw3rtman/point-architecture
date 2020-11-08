Executable = run_test.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_il_1.log
Output=/u/nimit/logs/$(ClusterId)_il_1.out
Error=/u/nimit/logs/$(ClusterId)_il_1.err

Queue 1
