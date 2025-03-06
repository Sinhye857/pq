#pragma once
#include <Eigen/Dense>

class SimilarityComputer {
public:
    // 用户相似度（余弦）
    Eigen::MatrixXf user_similarity(const Eigen::MatrixXf& user_matrix);
    
    // 商品相似度（转置后余弦）
    Eigen::MatrixXf item_similarity(const Eigen::MatrixXf& item_matrix);
    
    // 皮尔逊相关系数
    Eigen::MatrixXf pearson_correlation(const Eigen::MatrixXf& ratings_matrix);

private:
    void mean_centering(Eigen::MatrixXf& matrix);
};