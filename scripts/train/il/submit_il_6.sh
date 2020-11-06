Executable = run_il_6.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_il_6.log
Output=/u/nimit/logs/$(ClusterId)_il_6.out
Error=/u/nimit/logs/$(ClusterId)_il_6.err

Queue 1
