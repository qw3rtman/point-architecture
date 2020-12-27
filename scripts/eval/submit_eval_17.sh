Executable = run_eval_17.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_eval_17.log
Output=/u/nimit/logs/$(ClusterId)_eval_17.out
Error=/u/nimit/logs/$(ClusterId)_eval_17.err

Queue 1
