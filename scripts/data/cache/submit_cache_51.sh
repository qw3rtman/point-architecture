Executable = run_cache_51.sh

+Group="GRAD"
+Project="AI_ROBOTICS"
+ProjectDescription="Training model"
+GPUJob=true

Requirements=(TARGET.GPUSlot)
Rank=memory
Universe=vanilla
Getenv=True
Notification=Complete

Log=/u/nimit/logs/$(ClusterId)_cache_51.log
Output=/u/nimit/logs/$(ClusterId)_cache_51.out
Error=/u/nimit/logs/$(ClusterId)_cache_51.err

Queue 1
