cmake_minimum_required(VERSION 3.8)
project(vla_interfaces)

# ament_cmake を最初に見つける
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Anomaly.msg"
  "msg/Action.msg"
)

# ランタイム依存関係をエクスポート
ament_export_dependencies(rosidl_default_runtime)

ament_package()