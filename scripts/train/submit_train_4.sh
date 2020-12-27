Executable = run_train_4.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_train_4.log
Output=/u/nimit/logs/$(ClusterId)_train_4.out
Error=/u/nimit/logs/$(ClusterId)_train_4.err

Queue 1
