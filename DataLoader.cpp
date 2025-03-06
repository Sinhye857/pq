#include "DataLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;

void DataLoader::load(const string& path) {
    ifstream file(path);
    if (!file.is_open()) {
        throw runtime_error("无法打开文件: " + path);
    }

    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        int user_id, item_id;
        float weight;
        
        if (!(iss >> user_id >> item_id >> weight)) {
            continue; // 跳过格式错误行
        }

        // 建立用户映射
        if (!user_id_map_.count(user_id)) {
            user_id_map_[user_id] = user_item_matrix_.size();
            user_item_matrix_.emplace_back(item_dimension_, 0.0f);
        }

        // 建立商品映射
        if (!item_id_map_.count(item_id)) {
            item_id_map_[item_id] = item_id_map_.size();
        }

        // 填充交互矩阵（处理隐式反馈）
        size_t user_idx = user_id_map_[user_id];
        size_t item_idx = item_id_map_[item_id];
        if (item_idx < item_dimension_) {
            user_item_matrix_[user_idx][item_idx] = weight;
        }
    }
}

const vector<vector<float>>& DataLoader::get_user_item_matrix() const {
    return user_item_matrix_;
}

size_t DataLoader::get_item_dimension() const {
    return item_dimension_;
}