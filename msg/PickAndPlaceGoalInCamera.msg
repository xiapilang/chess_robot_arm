# chess_robot_arm/msg/PickAndPlaceGoalInCamera.msg
string object_id_at_pick    # 可选，抓取对象的ID (格式: "row,col" 或 "row,col,piece_type")
geometry_msgs/Point pick_position_in_camera   # 抓取点在基座标系中的位置 (x,y,z)

string target_location_id_at_place # 可选，放置位置的ID (格式: "row,col")
geometry_msgs/Point place_position_in_camera  # 放置点在基座标系中的位置 (x,y,z)

string piece_type             # 可选，被抓棋子的 FEN 符号 (p/n/b/r/q/k/P/N/B/R/Q/K)
