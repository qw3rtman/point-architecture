Executable = run_cache_21.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_cache_21.log
Output=/u/nimit/logs/$(ClusterId)_cache_21.out
Error=/u/nimit/logs/$(ClusterId)_cache_21.err

Queue 1
