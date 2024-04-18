# 指定 mingw
xmake f -p mingw
# 构建
xmake build hello_world
xmake build possion_editing
# 测试
xmake run hello_world
xmake run possion_editing

# 做 possion_editing 只需要以下几条
# 	xmake -r
#   xmake run possion_editing