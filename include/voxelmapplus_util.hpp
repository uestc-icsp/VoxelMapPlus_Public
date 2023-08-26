#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include "common_lib.h"
#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <cmath>
#include <random>

#define HASH_P 116101
#define MAX_N 10000000000

/*** Common Param ***/
static int plane_id = 0;
static int update_size_threshold;
static int max_points_size;
static int sigma_num;
static double planer_threshold;
static double voxel_size;
static double quater_length;

/*** Point to Plane Matching Structure ***/
typedef struct ptpl {
    V3D point;
    V3D point_world;
    V3D omega;
    double omega_norm = 0;
    double dist = 0;
    M3D plane_cov;
    int main_direction = 0;
} ptpl;

/*** 3D Point with Covariance ***/
typedef struct pointWithCov {
    V3D point;
    V3D point_world;
    Eigen::Matrix3d cov;
} pointWithCov;

/*** Plane Structure ***/
typedef struct Plane {
    /*** Update Flag ***/
    bool is_plane = false;
    bool is_init = false;
    
    /*** Plane Param ***/
    int main_direction = 0; //0:ax+by+z+d=0;  1:ax+y+bz+d=0;  2:x+ay+bz+d=0;
    M3D plane_cov;
    V3D n_vec;
    bool isRootPlane = true;
    int rgb[3] = {0, 0, 0};

    /*** Incremental Calculation Param ***/
    double xx = 0.0;
    double yy = 0.0;
    double zz = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yz = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    V3D center = V3D::Zero();
    Eigen::Matrix3d covariance = M3D::Zero();
    int points_size = 0;

} Plane;
typedef std::shared_ptr<Plane> PlanePtr;
typedef const std::shared_ptr<Plane> PlaneConstPtr;

class VOXEL_LOC {
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx, int64_t vy, int64_t vz)
            : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

// Hash value
namespace std {
    template<>
    struct hash<VOXEL_LOC> {
        int64_t operator()(const VOXEL_LOC &s) const {
            using std::hash;
            using std::size_t;
            return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        }
    };
} // namespace std

class UnionFindNode {
public:
    std::vector<pointWithCov> temp_points_; // all points in an octo tree
    PlanePtr plane_ptr_;
    double voxel_center_[3]{}; // x, y, z
    int all_points_num_;
    int new_points_num_;
    
    bool init_node_;
    bool update_enable_;
    bool is_plane;
    int id;
    UnionFindNode *rootNode;

    UnionFindNode(){
        temp_points_.clear();
        new_points_num_ = 0;
        all_points_num_ = 0;
        init_node_ = false;
        update_enable_ = true;
        plane_ptr_ = std::make_shared<Plane>();
        /*** Visualization Set RootNode Color ***/
        //初始化颜色
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 255);
        plane_ptr_->rgb[0] = dis(gen);
        plane_ptr_->rgb[1] = dis(gen);
        plane_ptr_->rgb[2] = dis(gen);
        rootNode = this;
    }

    /*** Finish ***/
    void InitPlane(const std::vector<pointWithCov> &points, const PlanePtr& plane, UnionFindNode* node) const {
        /*** Incremental Calculation about Covariance and SigmaXX ***/
        V3D last_center = plane->center;
        int last_size = plane->points_size;
        M3D last_EX2 = plane->covariance + last_center * last_center.transpose();
        M3D now_EX2 = last_EX2 * last_size;
        V3D now_center = last_center * last_size;
        for (int i = 0; i < new_points_num_; ++i) {
            now_center += points[last_size + i].point;
            now_EX2 += points[last_size + i].point * points[last_size + i].point.transpose();
            plane->xx += points[last_size + i].point[0] * points[last_size + i].point[0];
            plane->yy += points[last_size + i].point[1] * points[last_size + i].point[1];
            plane->zz += points[last_size + i].point[2] * points[last_size + i].point[2];
            plane->xy += points[last_size + i].point[0] * points[last_size + i].point[1];
            plane->xz += points[last_size + i].point[0] * points[last_size + i].point[2];
            plane->yz += points[last_size + i].point[1] * points[last_size + i].point[2];
            plane->x += points[last_size + i].point[0];
            plane->y += points[last_size + i].point[1];
            plane->z += points[last_size + i].point[2];
        }
        now_center = now_center / points.size();
        now_EX2 = now_EX2 / points.size();
        M3D now_cov = now_EX2 - now_center * now_center.transpose();
        plane->center = now_center;
        plane->covariance = now_cov;
        plane->points_size = static_cast<int>(points.size());

        /*** Plane Judgement based on Eigen Decomposition ***/
        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors();
        Eigen::Vector3cd evals = es.eigenvalues();
        V3D evalsReal;
        evalsReal = evals.real();
        Eigen::Matrix3f::Index evalsMin, evalsMax;
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = static_cast<int>(3 - evalsMin - evalsMax);
        V3D evecMin = evecs.real().col(evalsMin);

        /*** Definition ***/
        plane->plane_cov = M3D::Zero();
        double xx = plane->xx;
        double yy = plane->yy;
        double zz = plane->zz;
        double xy = plane->xy;
        double xz = plane->xz;
        double yz = plane->yz;
        double x = plane->x;
        double y = plane->y;
        double z = plane->z;
        int n = plane->points_size;
        double detA = 0.0;
        V3D E, ddetA_dpw;
        M3D dAstarE_dpw, J_pw, A_star, A;

        /*** 3DOF Plane Covariance Propagation ***/
        if (evalsReal(evalsMin) < planer_threshold) {
            //is plane
            if (fabs(evecMin[0]) >= fabs(evecMin[1]) && fabs(evecMin[0]) >= fabs(evecMin[2])) {
                //main_direction:2, x+ay+bz+d=0
                plane->main_direction = 2;
                E << -1.0 * xy, -1.0 * xz, -1.0 * x;
                A << yy, yz, y, yz, zz, z, y, z, n;
                detA = A.determinant();
                adjugateM3D(A, A_star);
                plane->n_vec = A_star * E / detA;
                for (auto pv: points) {
                    double xi = pv.point[0];
                    double yi = pv.point[1];
                    double zi = pv.point[2];
                    ddetA_dpw << 0.0,
                            2 * n * yi * zz + 2 * z * (yz + zi * y) - 2 * zz * y - 2 * yi * z * z - 2 * n * zi * yz,
                            2 * n * zi * yy + 2 * y * (yz + yi * z) - 2 * zi * y * y - 2 * yy * z - 2 * n * yi * yz;
                    dAstarE_dpw << yi * z * z - n * yi * zz + n * zi * yz - zi * y * z + zz * y - yz * z,
                            xi * z * z - n * xi * zz + n * zi * xz - xz * z + x * zz - zi * x * z,
                            2 * xy * z - 2 * n * zi * xy + n * (xi * yz + yi * xz) - y * (xz + xi * z) +
                            2 * zi * x * y - x * (yz + yi * z),
                            n * yi * yz - yi * y * z + zi * y * y - n * zi * yy + yy * z - yz * y,
                            n * (xi * yz + zi * xy) - z * (xy + xi * y) + 2 * xz * y - 2 * n * yi * xz +
                            2 * yi * x * z - x * (yz + zi * y),
                            n * yi * xy - xy * y + xi * y * y - n * xi * yy + x * yy - yi * x * y,
                            yi * zz * y - yi * yz * z + zi * yy * z - zi * yz * y + yz * yz - yy * zz,
                            zz * (xy + xi * y) - z * (xi * yz + zi * xy) + 2 * yi * xz * z - xz * (yz + zi * y) +
                            2 * zi * x * yz - 2 * yi * x * zz,
                            2 * zi * xy * y - xy * (yz + yi * z) + yy * (xz + xi * z) - y * (xi * yz + yi * xz) +
                            2 * yi * x * yz - 2 * zi * x * yy;
                    J_pw = A_star * E * (-1.0 * ddetA_dpw / detA / detA).transpose() + dAstarE_dpw / detA;
                    plane->plane_cov += J_pw * pv.cov * J_pw.transpose();
                }
            } else if (fabs(evecMin[1]) >= fabs(evecMin[0]) && fabs(evecMin[1]) >= fabs(evecMin[2])) {
                //main_direction:1, ax+y+bz+d=0
                plane->main_direction = 1;
                E << -1.0 * xy, -1.0 * yz, -1.0 * y;
                A << xx, xz, x, xz, zz, z, x, z, n;
                detA = A.determinant();
                adjugateM3D(A, A_star);
                plane->n_vec = A_star * E / detA;
                for (auto pv: points) {
                    double xi = pv.point[0];
                    double yi = pv.point[1];
                    double zi = pv.point[2];
                    ddetA_dpw
                            << 2 * n * xi * zz + 2 * z * (xz + zi * x) - 2 * zz * x - 2 * xi * z * z - 2 * n * zi * xz,
                            0.0,
                            2 * n * zi * xx + 2 * x * (xz + xi * z) - 2 * zi * x * x - 2 * xx * z - 2 * n * xi * xz;
                    dAstarE_dpw << yi * z * z - n * yi * zz + n * zi * yz - yz * z + y * zz - zi * y * z,
                            xi * z * z - n * xi * zz + n * zi * xz - zi * x * z + zz * x - xz * z,
                            2 * xy * z - 2 * n * zi * xy + n * (yi * xz + xi * yz) - x * (yz + yi * z) +
                            2 * zi * y * x - y * (xz + xi * z),
                            n * (yi * xz + zi * xy) - z * (xy + yi * x) + 2 * yz * x - 2 * n * xi * yz +
                            2 * xi * y * z - y * (xz + zi * x),
                            n * xi * xz - xi * x * z + zi * x * x - n * zi * xx + xx * z - xz * x,
                            n * xi * xy - xy * x + yi * x * x - n * yi * xx + y * xx - xi * y * x,
                            zz * (yi * x + xy) - z * (yi * xz + zi * xy) + 2 * xi * yz * z - yz * (xz + zi * x) +
                            2 * zi * y * xz - 2 * xi * y * zz,
                            xi * zz * x - xi * xz * z + zi * xx * z - zi * xz * x + xz * xz - xx * zz,
                            2 * zi * xy * x - xy * (xi * z + xz) + xx * (yz + yi * z) - x * (yi * xz + xi * yz) +
                            2 * xi * y * xz - 2 * zi * y * xx;
                    J_pw = A_star * E * (-1.0 * ddetA_dpw / detA / detA).transpose() + dAstarE_dpw / detA;
                    plane->plane_cov += J_pw * pv.cov * J_pw.transpose();
                }
            } else {
                //main_direction:0, ax+by+z+d=0
                plane->main_direction = 0;
                A << xx, xy, x, xy, yy, y, x, y, n;
                E << -1.0 * xz, -1.0 * yz, -1.0 * z;
                detA = A.determinant();
                adjugateM3D(A, A_star);
                plane->n_vec = A_star * E / detA;
                for (auto pv: points) {
                    double xi = pv.point[0];
                    double yi = pv.point[1];
                    double zi = pv.point[2];
                    ddetA_dpw
                            << 2 * n * xi * yy + 2 * y * (xy + yi * x) - 2 * yy * x - 2 * xi * y * y - 2 * n * yi * xy,
                            2 * n * yi * xx + 2 * x * (xy + xi * y) - 2 * xx * y - 2 * yi * x * x - 2 * n * xi * xy,
                            0.0;
                    dAstarE_dpw << zi * y * y - n * zi * yy + n * yi * yz - yz * y + yy * z - yi * y * z,
                            2 * xz * y - 2 * n * yi * xz + n * (xi * yz + zi * xy) - x * (yz + zi * y) +
                            2 * yi * x * z - z * (xy + xi * y),
                            xi * y * y - n * xi * yy + n * yi * xy - yi * x * y + yy * x - xy * y,
                            n * (yi * xz + zi * xy) - y * (xz + zi * x) + 2 * yz * x - 2 * n * xi * yz +
                            2 * xi * y * z - z * (yi * x + xy),
                            n * xi * xz - x * xz + zi * x * x - n * zi * xx + xx * z - xi * x * z,
                            n * xi * xy - xi * x * y + yi * x * x - n * yi * xx + xx * y - xy * x,
                            yy * (xz + zi * x) - y * (zi * xy + yi * xz) + 2 * xi * yz * y - yz * (xy + yi * x) +
                            2 * yi * z * xy - 2 * xi * z * yy,
                            2 * yi * xz * x - xz * (xi * y + xy) + xx * (yz + zi * y) - x * (zi * xy + xi * yz) +
                            2 * xi * z * xy - 2 * yi * z * xx,
                            xi * yy * x - xi * xy * y + yi * xx * y - yi * xy * x + xy * xy - xx * yy;
                    J_pw = A_star * E * (-1.0 * ddetA_dpw / detA / detA).transpose() + dAstarE_dpw / detA;
                    plane->plane_cov += J_pw * pv.cov * J_pw.transpose();
                }
            }
            plane->is_plane = true;
            node->is_plane = true;
            if (!plane->is_init) {
                node->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }
        } else {//is not plane
            if (!plane->is_init) {
                node->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }
            plane->is_plane = false;
            node->is_plane = false;
        }
    }

    void InitUnionFindNode() {
        if (temp_points_.size() > update_size_threshold) {
            //init_plane(temp_points_, plane_ptr_);
            InitPlane(temp_points_, plane_ptr_, this);
            if (plane_ptr_->is_plane) {
                if (temp_points_.size() > max_points_size) {
                    update_enable_ = false;
                }
            }
            init_node_ = true;
            new_points_num_ = 0;
            //      temp_points_.clear();
        }
    }

    void UpdatePlane(const pointWithCov &pv,
                     VOXEL_LOC &position, std::unordered_map<VOXEL_LOC, UnionFindNode *> &feat_map) {
        if (!init_node_) {
            new_points_num_++;
            all_points_num_++;
            temp_points_.push_back(pv);
            if (temp_points_.size() > update_size_threshold) {
                InitUnionFindNode();
            }
        } else {
            if (is_plane) {
                if (update_enable_) {
                    //点数在1000以下则与VoxelMap相同(这里可以忽略不看)
                    new_points_num_++;
                    all_points_num_++;
                    if (update_enable_) {
                        temp_points_.push_back(pv);
                    }
                    if (new_points_num_ > update_size_threshold) {
                        if (update_enable_) {
                            InitPlane(temp_points_, plane_ptr_, this);
                        }
                        new_points_num_ = 0;
                    }
                    if (all_points_num_ >= max_points_size) {
                        update_enable_ = false;
                        std::vector<pointWithCov>().swap(temp_points_);
                    }
                } else {
                    // 点数已经足够，可实行融合
                    // nowRealRootNode: realRootNode
                    UnionFindNode *nowRealRootNode = this;
                    // 寻找当前节点的根节点
                    // 本while也许可以直接删除
                    while (nowRealRootNode != nowRealRootNode->rootNode) {
                        nowRealRootNode = nowRealRootNode->rootNode;
                    }

                    for (int k = 0; k < 6; k++) {
                        // 检查周围6个节点
                        switch (k) {
                            case 0:
                                position.x = position.x - 1;
                                break;
                            case 1:
                                position.x = position.x + 2;
                                break;
                            case 2:
                                position.x = position.x - 1;
                                position.y = position.y - 1;
                                break;
                            case 3:
                                position.y = position.y + 2;
                                break;
                            case 4:
                                position.y = position.y - 1;
                                position.z = position.z + 1;
                                break;
                            case 5:
                                position.z = position.z - 2;
                                break;
                            default:
                                break;
                        }
                        auto iter = feat_map.find(position);
                        if (iter != feat_map.end()) {
                            //neighbor_plane所在的octotree可能不是root,所以要找到它的根节点
                            UnionFindNode *neighRealRootNode = iter->second;
                            //找邻居的根节点
                            while (neighRealRootNode != neighRealRootNode->rootNode) {
                                neighRealRootNode = neighRealRootNode->rootNode;
                            }
                            //邻居与当前平面可能是相同root或点数不够
                            if (neighRealRootNode == nowRealRootNode || neighRealRootNode->update_enable_) {
                                continue;
                            }
                            PlanePtr neighbor_plane = neighRealRootNode->plane_ptr_;
                            PlanePtr now_plane = nowRealRootNode->plane_ptr_;
                            /*** Plane Merging ***/
                            if (neighbor_plane->is_plane) {
                                if (neighbor_plane->main_direction == now_plane->main_direction) {
                                    V3D abd_bias = (neighbor_plane->n_vec - now_plane->n_vec).cwiseAbs();
                                    double m_distance = sqrt(abd_bias.transpose() *
                                            (neighbor_plane->plane_cov + now_plane->plane_cov).inverse() * abd_bias);
                                    if ((abd_bias[0] < 0.1 && abd_bias[1] < 0.1) || m_distance < 0.004) {
                                        neighRealRootNode->rootNode = nowRealRootNode;
                                        neighbor_plane->isRootPlane = false;
                                        double paraA = neighbor_plane->plane_cov.norm() /
                                                       (nowRealRootNode->plane_ptr_->plane_cov.norm() +
                                                        neighbor_plane->plane_cov.norm());
                                        double paraB = nowRealRootNode->plane_ptr_->plane_cov.norm() /
                                                       (nowRealRootNode->plane_ptr_->plane_cov.norm() +
                                                        neighbor_plane->plane_cov.norm());
                                        nowRealRootNode->plane_ptr_->n_vec =
                                                paraA * nowRealRootNode->plane_ptr_->n_vec +
                                                paraB * neighbor_plane->n_vec;
                                        nowRealRootNode->plane_ptr_->plane_cov =
                                                paraA * paraA * nowRealRootNode->plane_ptr_->plane_cov +
                                                paraB * paraB * neighbor_plane->plane_cov;
                                    }
                                }
                            }
                        }
                    }
                    /*return;*/
                }
            } else {
                if (update_enable_) {
                    new_points_num_++;
                    all_points_num_++;
                    if (update_enable_) {
                        temp_points_.push_back(pv);
                    }
                    if (new_points_num_ > update_size_threshold) {
                        if (update_enable_) {
                            InitPlane(temp_points_, plane_ptr_, this);
                        }
                        new_points_num_ = 0;
                    }
                    if (all_points_num_ >= max_points_size) {
                        update_enable_ = false;
                        std::vector<pointWithCov>().swap(temp_points_);
                    }
                }
            }
        }
    }

};

void MapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin) {
        v = vmin;
    }

    if (v > vmax) {
        v = vmax;
    }

    double dr, dg, db;

    if (v < 0.1242) {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    } else if (v < 0.3747) {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    } else if (v < 0.6253) {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    } else if (v < 0.8758) {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    } else {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t) (255 * dr);
    g = (uint8_t) (255 * dg);
    b = (uint8_t) (255 * db);
}

void BuildVoxelMap(const std::vector<pointWithCov> &input_points, 
                   std::unordered_map<VOXEL_LOC, UnionFindNode *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov& p_v = input_points[i];
        double loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1],
                           (int64_t) loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
        } else {
            auto *octo_tree = new UnionFindNode();
            feat_map[position] = octo_tree;
            feat_map[position]->voxel_center_[0] = (0.5 + static_cast<double>(position.x)) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + static_cast<double>(position.y)) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + static_cast<double>(position.z)) * voxel_size;
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
        }
    }
    for (auto & iter : feat_map) {
        iter.second->InitUnionFindNode();
    }
}

void UpdateVoxelMap(const std::vector<pointWithCov> &input_points, 
                    std::unordered_map<VOXEL_LOC, UnionFindNode *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov& p_v = input_points[i];
        double loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1],
                           (int64_t) loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) { //这个是第0层
            feat_map[position]->UpdatePlane(p_v, position, feat_map);
        } else {
            auto *node = new UnionFindNode();
            feat_map[position] = node;
            feat_map[position]->voxel_center_[0] = (0.5 + static_cast<double>(position.x)) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + static_cast<double>(position.y)) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + static_cast<double>(position.z)) * voxel_size;
            feat_map[position]->UpdatePlane(p_v, position, feat_map);
        }
    }
}

void TransformLidar(const StatesGroup &state,
                    const shared_ptr<ImuProcess> &p_imu,
                    const PointCloudXYZI::Ptr &input_cloud,
                    pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud) {
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        V3D p(p_c.x, p_c.y, p_c.z);
        p = state.rot_end * p + state.pos_end;
        pcl::PointXYZI pi;
        pi.x = static_cast<float>(p(0));
        pi.y = static_cast<float>(p(1));
        pi.z = static_cast<float>(p(2));
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

void BuildSingleResidual(const pointWithCov &pv, const UnionFindNode *currentNode,
                         bool &is_sucess, ptpl &single_ptpl) {
    V3D p_w = pv.point_world;
    if (currentNode->plane_ptr_->is_plane) {
        Plane &plane = *currentNode->plane_ptr_;
        V3D p_world_to_center = p_w - plane.center;
        Eigen::Matrix<double, 1, 3> J_abd = Eigen::Matrix<double, 1, 3>::Zero();
        Eigen::Matrix<double, 1, 3> J_pw = Eigen::Matrix<double, 1, 3>::Zero();
        /*** Corresponding Plane Registration ***/
        single_ptpl.point = pv.point;
        single_ptpl.plane_cov = plane.plane_cov;
        single_ptpl.main_direction = plane.main_direction;
        single_ptpl.omega_norm = sqrt(plane.n_vec[0] * plane.n_vec[0] + plane.n_vec[1] * plane.n_vec[1] + 1);
        single_ptpl.point_world = pv.point_world;
        /*** Distance Covariance Propagation ***/
        double omega_norm_2 = single_ptpl.omega_norm * single_ptpl.omega_norm;
        switch (plane.main_direction) {
            case 0:
                single_ptpl.omega << plane.n_vec[0], plane.n_vec[1], 1;
                single_ptpl.dist = (pv.point_world.transpose() * single_ptpl.omega + plane.n_vec[2]) /
                                   single_ptpl.omega_norm;
                J_abd << pv.point_world(0) * (1 - single_ptpl.dist / omega_norm_2),
                        pv.point_world(1) * (1 - single_ptpl.dist / omega_norm_2), 1;
                break;
            case 1:
                single_ptpl.omega << plane.n_vec[0], 1, plane.n_vec[1];
                single_ptpl.dist = (pv.point_world.transpose() * single_ptpl.omega + plane.n_vec[2]) /
                                   single_ptpl.omega_norm;
                J_abd << pv.point_world(0) * (1 - single_ptpl.dist / omega_norm_2),
                        pv.point_world(2) * (1 - single_ptpl.dist / omega_norm_2), 1;
                break;
            case 2:
                single_ptpl.omega << 1, plane.n_vec[0], plane.n_vec[1];
                single_ptpl.dist = (pv.point_world.transpose() * single_ptpl.omega + plane.n_vec[2]) /
                                   single_ptpl.omega_norm;
                J_abd << pv.point_world(1) * (1 - single_ptpl.dist / omega_norm_2),
                        pv.point_world(2) * (1 - single_ptpl.dist / omega_norm_2), 1;
                break;
        }
        J_pw =  single_ptpl.omega.transpose() /  single_ptpl.omega_norm;
        double sigma_l = J_abd * plane.plane_cov * J_abd.transpose();
        sigma_l += J_pw * pv.cov * J_pw.transpose();
        /*** 3 Sigma Outlier Remove ***/
        if (single_ptpl.dist < sigma_num * sqrt(sigma_l)) {
            is_sucess = true;
        } else {
            is_sucess = false;
        }
    }
}

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<V3D> &non_match) {
    std::mutex mylock;
    ptpl_list.clear();
    std::vector<ptpl> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size());
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
        useful_ptpl[i] = false;
    }
//#ifdef MP_EN
//    omp_set_num_threads(MP_PROC_NUM);
//#pragma omp parallel for
//#endif
    for (int i = 0; i < index.size(); i++) {
        const pointWithCov& pv = pv_list[i];
        double loc_xyz[3];
        for (int j = 0; j < 3; j++) {
            loc_xyz[j] = pv.point_world[j] / voxel_size;
            if (loc_xyz[j] < 0) {
                loc_xyz[j] -= 1.0;
            }
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1],
                           (int64_t) loc_xyz[2]);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            UnionFindNode *currentRootNode = iter->second;
            ptpl single_ptpl;
            bool is_sucess = false;

            /*** Pruning and Memory Free***/
            while (currentRootNode->rootNode != currentRootNode) {
                currentRootNode = currentRootNode->rootNode;
                if(currentRootNode->rootNode == currentRootNode){
                    iter->second->rootNode = currentRootNode;
                    iter->second->plane_ptr_.reset();
                    // use_count = 0, free the plane memory usage
                }
            }

            BuildSingleResidual(pv, currentRootNode, is_sucess, single_ptpl);
            if (!is_sucess) {
                VOXEL_LOC near_position = position;
                if (loc_xyz[0] > (currentRootNode->voxel_center_[0] + quater_length)) {
                    near_position.x = near_position.x + 1;
                } else if (loc_xyz[0] < (currentRootNode->voxel_center_[0] - quater_length)) {
                    near_position.x = near_position.x - 1;
                }
                if (loc_xyz[1] > (currentRootNode->voxel_center_[1] + quater_length)) {
                    near_position.y = near_position.y + 1;
                } else if (loc_xyz[1] < (currentRootNode->voxel_center_[1] - quater_length)) {
                    near_position.y = near_position.y - 1;
                }
                if (loc_xyz[2] > (currentRootNode->voxel_center_[2] + quater_length)) {
                    near_position.z = near_position.z + 1;
                } else if (loc_xyz[2] < (currentRootNode->voxel_center_[2] - quater_length)) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end()) {
                    UnionFindNode *near_octo = iter_near->second;
                    while (near_octo->rootNode != near_octo) {
                        near_octo = near_octo->rootNode;
                    }
                    BuildSingleResidual(pv, near_octo, is_sucess, single_ptpl);
                }
            }
            if (is_sucess) {
                mylock.lock();
                useful_ptpl[i] = true;
                all_ptpl_list[i] = single_ptpl;
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[i] = false;
                mylock.unlock();
            }
        }
    }
    for (size_t i = 0; i < useful_ptpl.size(); i++) {
        if (useful_ptpl[i]) {
            ptpl_list.push_back(all_ptpl_list[i]);
        } else {
            non_match.push_back(all_ptpl_list[i].point);
        }
    }
}

/*** Visualization Function ***/
void GetUpdatePlane(const UnionFindNode *current_octo, std::vector<Plane> &plane_list) {
    if (current_octo->all_points_num_ >= 100) {
        plane_list.push_back(*current_octo->plane_ptr_);
    }
}

/*** Visualization Function ***/
void CalcVectQuaternion(const Plane &single_plane, geometry_msgs::Quaternion &q) {
    //int main_direction = 0; //0:ax+by+z+d=0;  1:ax+y+bz+d=0;  2:x+ay+bz+d=0;
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    if (single_plane.main_direction == 0) {
        a = single_plane.n_vec[0];
        b = single_plane.n_vec[1];
        c = 1;
    } else if (single_plane.main_direction == 1) {
        a = single_plane.n_vec[0];
        b = 1.0;
        c = single_plane.n_vec[1];
    } else if (single_plane.main_direction == 2) {
        a = 1;
        b = single_plane.n_vec[0];
        c = single_plane.n_vec[1];
    }
    double t1 = sqrt(a * a + b * b + c * c);
    a = a / t1;
    b = b / t1;
    c = c / t1;
    double theta_half = acos(c) / 2;
    double t2 = sqrt(a * a + b * b);
    b = b / t2;
    a = a / t2;
    q.w = cos(theta_half);
    q.x = b * sin(theta_half);
    q.y = -1 * a * sin(theta_half);
    q.z = 0.0;

}

/*** Visualization Function ***/
void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string& plane_ns, const Plane &single_plane,
                    const float alpha, const V3D& rgb, int id) {
    visualization_msgs::Marker plane;
    plane.header.frame_id = "camera_init";
    plane.header.stamp = ros::Time();
    plane.ns = plane_ns;
    plane.id = id;
    // todo: change Type
    if (single_plane.isRootPlane) {
        plane.type = visualization_msgs::Marker::CYLINDER;
    } else {
        plane.type = visualization_msgs::Marker::CUBE;
    }

    plane.action = visualization_msgs::Marker::ADD;
    // todo:重新计算参数

    plane.pose.position.x = single_plane.center[0];
    plane.pose.position.y = single_plane.center[1];
    plane.pose.position.z = single_plane.center[2];
    geometry_msgs::Quaternion q;
    CalcVectQuaternion(single_plane, q);
    plane.pose.orientation = q;
    plane.scale.x = 0.45;
    plane.scale.y = 0.45;
    plane.scale.z = 0.01;
    plane.color.a = alpha;
    plane.color.r = static_cast<float>(rgb(0));
    plane.color.g = static_cast<float>(rgb(1));
    plane.color.b = static_cast<float>(rgb(2));
    plane.lifetime = ros::Duration();
    plane_pub.markers.push_back(plane);
}

/*** Visualization Function ***/
void pubVoxelMap(const std::unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map,
                 const ros::Publisher &plane_map_pub) {
    double max_trace = 0.25;
    double pow_num = 0.2;
    ros::Rate loop(500);
    float use_alpha = 0.8;
    visualization_msgs::MarkerArray voxel_plane;
    voxel_plane.markers.reserve(1000000);
    std::vector<UnionFindNode *> pub_node_list;
    for (const auto & iter : voxel_map) {
        if (!iter.second->update_enable_) {
            pub_node_list.emplace_back(iter.second);
        }
    }
    for (auto & node : pub_node_list) {
        UnionFindNode *curRootNode = node;
        while (curRootNode->rootNode != curRootNode) {
            curRootNode = curRootNode->rootNode;
        }

        V3D plane_rgb(curRootNode->plane_ptr_->rgb[0] / 256.0,
                      curRootNode->plane_ptr_->rgb[1] / 256.0,
                      curRootNode->plane_ptr_->rgb[2] / 256.0);
        float alpha;
        if (curRootNode->is_plane) {
            alpha = use_alpha;

            Plane newP;
            if(node == curRootNode) {
                newP.isRootPlane = true;
            }else{
                newP.isRootPlane = false;
            }
            newP.n_vec = curRootNode->plane_ptr_->n_vec;
            newP.center[0] = node->voxel_center_[0];
            newP.center[1] = node->voxel_center_[1];
            newP.center[2] = node->voxel_center_[2];
            newP.main_direction = curRootNode->plane_ptr_->main_direction;
            pubSinglePlane(voxel_plane, "plane", newP, alpha, plane_rgb, node->id);
        } else {
            alpha = 0;
        }
    }
    std::cout << "voxel_plane size:" << voxel_plane.markers.size() << std::endl;
    plane_map_pub.publish(voxel_plane);
    loop.sleep();
}

void calcBodyCov(V3D &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov) {
    double range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
    double range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
            pow(sin(DEG2RAD(degree_inc)), 2);
    V3D direction(pb);
    // 防止NAN
    if (direction(2) == 0) {
        direction(2) = 1e-6;
    }
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0,
            -direction(0), -direction(1), direction(0), 0;
    V3D base_vector1(1, 1,
                     -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    V3D base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
            base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() +
          A * direction_var * A.transpose();
}

#endif
