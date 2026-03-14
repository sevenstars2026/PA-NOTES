# PA1 NOTES
## 1.init_isa
实现客户程序映射到客户计算机：guest_to_host
客户计算机nemu的模拟内存pmem(physical memory)实际是host的uint8_t数组（*硬件内存最小单位1字节8位，模拟器需要严格按照字节模拟内存的逻辑，控制溢出一致*）
## 2.static 
限制此变量，只有本文件能访问，外部文件需要新建指针访问内存