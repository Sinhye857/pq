#pragma once
#include <vector>
#include <unordered_map>
#include <string>

class DataLoader {
public:
    void load(const std::string& path);
    const std::vector<std::vector<float>>& get_user_item_matrix() const;
    size_t get_item_dimension() const;
    
private:
    std::unordered_map<int, size_t> user_id_map_;
    std::unordered_map<int, size_t> item_id_map_;
    std::vector<std::vector<float>> user_item_matrix_;
    size_t item_dimension_ = 128;
};