from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# [name: "/device:CPU:0"
# device_type: "CPU"
# memory_limit: 268435456
# locality {
# }
# incarnation: 639368550075527088
# , name: "/device:GPU:0"
# device_type: "GPU"
# memory_limit: 2914163099
# locality {
#   bus_id: 1
#   links {
#   }
# }
# incarnation: 1921348525942958035
# physical_device_desc: "device: 0, name: NVIDIA GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1"
# ]
