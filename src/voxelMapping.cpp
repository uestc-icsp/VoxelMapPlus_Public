#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "voxelmapplus_util.hpp"
#include <Eigen/Core>
#include <common_lib.h>
#include <csignal>
#include <cv_bridge/cv_bridge.h>
#include <fstream>
#include <geometry_msgs/Vector3.h>
#include <image_transport/image_transport.h>
#include <livox_ros_driver/CustomMsg.h>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <opencv2/opencv.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf2_msgs/TFMessage.h>
#include <thread>
#include <unistd.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <voxel_map_plus/States.h>

#define INIT_TIME (0.0)
#define CALIB_ANGLE_COV (0.01)
bool calib_laser = false;
bool write_kitti_log = false;
std::string result_path = "";
// params for imu
bool imu_en = true;
std::vector<double> extrinT;
std::vector<double> extrinR;

// params for publish function
bool publish_voxel_map = false;
bool publish_point_cloud = false;
int pub_point_cloud_skip = 1;

// record point usage
double mean_effect_points = 0;
double mean_ds_points = 0;
double mean_raw_points = 0;

// record time
double undistort_time_mean = 0;
double down_sample_time_mean = 0;
double calc_cov_time_mean = 0;
double scan_match_time_mean = 0;
double ekf_solve_time_mean = 0;
double map_update_time_mean = 0;

mutex mtx_buffer;
condition_variable sig_buffer;
Eigen::Matrix3d last_rot = Eigen::Matrix3d::Zero();

string lid_topic, imu_topic;
int scanIdx = 0;

int iterCount, NUM_MAX_ITERATIONS, effct_feat_num, time_log_counter, publish_count = 0;

double first_lidar_time = 0;
double lidar_end_time = 0;
double res_mean_last = 0.05;
double total_distance = 0;
double gyr_cov_scale, acc_cov_scale;
double last_timestamp_lidar, last_timestamp_imu = -1.0;
double filter_size_surf_min;
double map_incremental_time, total_time, scan_match_time, solve_time;
bool lidar_pushed, flg_reset, flg_exit = false;
bool dense_map_en = true;

deque<PointCloudXYZI::Ptr> lidar_surf_buffer;
deque<double> time_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;


PointCloudXYZI::Ptr surf_feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudNoeffect(new PointCloudXYZI(100000, 1));
pcl::VoxelGrid<PointType> downSizeFilterSurf;

V3D euler_cur;
V3D position_last(Zero3d);

// estimator inputs and output;
MeasureGroup Measures;
StatesGroup state;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

void SigHandle(int sig) {
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

inline void kitti_log(FILE *fp) {
    Eigen::Matrix4d T_lidar_to_cam;
    T_lidar_to_cam << 0.00042768, -0.999967, -0.0080845, -0.01198, -0.00721062,
            0.0080811998, -0.99994131, -0.0540398, 0.999973864, 0.00048594,
            -0.0072069, -0.292196, 0, 0, 0, 1.0;
    V3D rot_ang(Log(state.rot_end));
    MD(4, 4) T;
    T.block<3, 3>(0, 0) = state.rot_end;
    T.block<3, 1>(0, 3) = state.pos_end;
    T(3, 0) = 0;
    T(3, 1) = 0;
    T(3, 2) = 0;
    T(3, 3) = 1;
    T = T_lidar_to_cam * T * T_lidar_to_cam.inverse();
    for (int i = 0; i < 3; i++) {
        if (i == 2)
            fprintf(fp, "%lf %lf %lf %lf", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
        else
            fprintf(fp, "%lf %lf %lf %lf ", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
    }
    fprintf(fp, "\n");
    fflush(fp);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state.rot_end * (p_body) + state.pos_end);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
    po->normal_y = pi->normal_y;
    po->normal_z = pi->normal_z;
    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity * 10000;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    mtx_buffer.lock();
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_surf_buffer.clear();
    }
    PointCloudXYZI::Ptr surf_ptr(new PointCloudXYZI());
    p_pre->process(msg, surf_ptr);
    lidar_surf_buffer.push_back(surf_ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 接收livox消息，加入buffer并开锁
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
    mtx_buffer.lock();
    // cout << "got feature" << endl;
    if (msg->header.stamp.toSec() < last_timestamp_lidar) {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_surf_buffer.clear();
    }
    PointCloudXYZI::Ptr surf_ptr(new PointCloudXYZI());
    p_pre->process(msg, surf_ptr);
    lidar_surf_buffer.push_back(surf_ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 接收imu消息，加入buffer并开锁
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
    publish_count++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));
    double timestamp = msg->header.stamp.toSec();
    mtx_buffer.lock();
    if (timestamp < last_timestamp_imu) {
        ROS_ERROR("imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// 将异步的imu与Lidar进行打包
// 打包一帧lidar以及本lidar_end_time之前的所有imu
bool sync_packages(MeasureGroup &meas) {

    // 无imu，则只需要打包lidar信息
    if (!imu_en) {
        if (!lidar_surf_buffer.empty()) {
            // cout<<"meas.lidar->points.size(): "<<meas.lidar->points.size()<<endl;
            meas.surf_lidar = lidar_surf_buffer.front();
            // meas.corn_lidar     = lidar_corn_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_surf_buffer.pop_front();
            // lidar_corn_buffer.pop_front();
            return true;
        }

        return false;
    }

    // 有imu
    if (lidar_surf_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.surf_lidar = lidar_surf_buffer.front();
        // meas.corn_lidar = lidar_corn_buffer.front();
        if (meas.surf_lidar->points.size() <= 1) {
            lidar_surf_buffer.pop_front();
            // lidar_corn_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        lidar_end_time = meas.lidar_beg_time +
                         meas.surf_lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        // 收集所有imu，直到imu时间 > lidar时间 0.02s
        if (imu_time > lidar_end_time + 0.02)
            break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_surf_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void publish_surf_frame_world(const ros::Publisher &pubLaserCloud,
                              const int point_skip) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? surf_feats_undistort
                                                       : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++) {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                            &laserCloudWorld->points[i]);
    }
    PointCloudXYZI::Ptr laserCloudWorldPub(new PointCloudXYZI);
    for (int i = 0; i < size; i += point_skip) {
        laserCloudWorldPub->points.push_back(laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorldPub, laserCloudmsg);
    laserCloudmsg.header.stamp =
            ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloud.publish(laserCloudmsg);
}

// publish有效点，Intensity由标准差着色
void publish_effect(const ros::Publisher &pubLaserCloudEffect) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud_world(
            new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++) {
        RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
        pcl::PointXYZRGB pi;
        pi.x = laserCloudWorld->points[i].x;
        pi.y = laserCloudWorld->points[i].y;
        pi.z = laserCloudWorld->points[i].z;
        float v = laserCloudWorld->points[i].intensity / 100;
        v = 1.0 - v;
        uint8_t r, g, b;
        MapJet(v, 0, 1, r, g, b);
        pi.r = r;
        pi.g = g;
        pi.b = b;
        effect_cloud_world->points.push_back(pi);
    }

    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp =
            ros::Time::now(); //.fromSec(last_timestamp_lidar);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template<typename T>
void set_posestamp(T &out) {
    out.position.x = state.pos_end(0);
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

void publish_odometry(const ros::Publisher &pubOdomAftMapped) {
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp =
            ros::Time::now(); // ros::Time().fromSec(last_timestamp_lidar);
    set_posestamp(odomAftMapped.pose.pose);
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(
            tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));
    q.setW(geoQuat.w);
    q.setX(geoQuat.x);
    q.setY(geoQuat.y);
    q.setZ(geoQuat.z);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp,
                                          "camera_init", "aft_mapped"));
    pubOdomAftMapped.publish(odomAftMapped);
}

void publish_path(const ros::Publisher pubPath) {
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath.publish(path);
}

// VoxelMap主函数入口，单节点应用
int main(int argc, char **argv) {
    ros::init(argc, argv, "voxelMapping");
    ros::NodeHandle nh;

    double ranging_cov = 0.0;
    double angle_cov = 0.0;
    std::vector<double> layer_point_size;

    // cummon params
    nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");

    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov_scale", gyr_cov_scale, 0.1);
    nh.param<double>("noise_model/acc_cov_scale", acc_cov_scale, 0.1);

    // imu params, current version does not support imu
    nh.param<bool>("imu/imu_en", imu_en, false);
    nh.param<vector<double>>("imu/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("imu/extrinsic_R", extrinR, vector<double>());

    // mapping algorithm params
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/update_size_threshold", update_size_threshold, 5);
    nh.param<int>("mapping/sigma_num", sigma_num, 3);
    nh.param<double>("mapping/voxel_size", voxel_size, 1.0);
    quater_length = voxel_size / 4;
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    nh.param<double>("mapping/plannar_threshold", planer_threshold, 0.01);

    // preprocess params
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<bool>("preprocess/calib_laser", calib_laser, false);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);

    // visualization params
    nh.param<bool>("visualization/pub_voxel_map", publish_voxel_map, false);
    std::cout << "publishe_voxel_map:" << publish_voxel_map << std::endl;


    nh.param<bool>("visualization/pub_point_cloud", publish_point_cloud, true);
    nh.param<int>("visualization/pub_point_cloud_skip", pub_point_cloud_skip, 1);
    nh.param<bool>("visualization/dense_map_enable", dense_map_en, false);

    // result params
    nh.param<bool>("Result/write_kitti_log", write_kitti_log, false);
    nh.param<string>("Result/result_path", result_path,
                     "/home/ycj/catkin_github/src/VoxelMapPlus/Log/kitt_log.txt");
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    // 接收Lidar消息
    ros::Subscriber sub_pcl =
            p_pre->lidar_type == AVIA
            ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk)
            : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu;
    // 接收IMU消息
    if (imu_en) {
        sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    }

    ros::Publisher pubLaserCloudSurfFull =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_surf", 100);
    ros::Publisher pubLaserCloudEffect =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
    ros::Publisher pubOdomAftMapped =
            nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
    ros::Publisher voxel_map_pub =
            nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

    path.header.stamp = ros::Time::now();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    VD(DIM_STATE) solution;
    MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
    V3D rot_add, t_add;
    StatesGroup state_propagat;
    PointType pointOri, pointSel, coeff;
    int frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0;
    bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0,
            is_first_frame = true;
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                   filter_size_surf_min);

    shared_ptr<ImuProcess> p_imu(new ImuProcess());
    p_imu->imu_en = imu_en;
    Eigen::Vector3d extT = V3D::Zero();
    Eigen::Matrix3d extR = M3D::Identity();
    extT << extrinT[0], extrinT[1], extrinT[2];
    extR << extrinR[0], extrinR[1], extrinR[2], extrinR[3], extrinR[4],
            extrinR[5], extrinR[6], extrinR[7], extrinR[8];
    p_imu->set_extrinsic(extT, extR);

    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));

    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    /*** debug record ***/
    FILE *fp_kitti;

    if (write_kitti_log) {
        fp_kitti = fopen(result_path.c_str(), "w");
    }

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();

    // for Plane Map
    bool init_map = false;
    std::unordered_map<VOXEL_LOC, UnionFindNode *> voxel_map;
    last_rot << 1, 0, 0, 0, 1, 0, 0, 0, 1;

    //TEST4
    bool enable_write = false;
    ofstream foutC;
    foutC.open("/home/yyf/ws4voxelmapplus/datas/s02_time.txt");

    while (status) {
        if (flg_exit) {
            break;
        }
        ros::spinOnce();

        /*** 1.Sync Package ***/
        if (sync_packages(Measures)) {
            //std::cout << "sync once" << std::endl;
            // 时间回溯的保护逻辑
            if (flg_reset) {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }

            std::cout << "scanIdx:" << scanIdx << std::endl;
            double t0, t1, t2, t3, t4, t5, match_start, match_time, solve_start,
                    svd_time;
            match_time = 0;
            solve_time = 0;
            svd_time = 0;


            /*** 2.IMU Predict ***/
            /*** 3.PointCloud Undistort ***/
            auto undistort_start = std::chrono::high_resolution_clock::now();
            p_imu->Process(Measures, state, surf_feats_undistort);
            auto undistort_end = std::chrono::high_resolution_clock::now();
            auto undistort_time =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                            undistort_end - undistort_start)
                            .count() *
                    1000;
            // only for kitti
            if (calib_laser) {
                // calib the vertical angle for kitti dataset
                for (size_t i = 0; i < surf_feats_undistort->size(); i++) {
                    PointType pi = surf_feats_undistort->points[i];
                    double range = sqrt(pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
                    double calib_vertical_angle = deg2rad(0.15);
                    double vertical_angle = asin(pi.z / range) + calib_vertical_angle;
                    double horizon_angle = atan2(pi.y, pi.x);
                    pi.z = range * sin(vertical_angle);
                    double project_len = range * cos(vertical_angle);
                    pi.x = project_len * cos(horizon_angle);
                    pi.y = project_len * sin(horizon_angle);
                    surf_feats_undistort->points[i] = pi;
                }
            }
            state_propagat = state;

            if (is_first_frame) {
                first_lidar_time = Measures.lidar_beg_time;
                is_first_frame = false;
            }

            if (surf_feats_undistort->empty() || (surf_feats_undistort == NULL)) {
                p_imu->first_lidar_time = first_lidar_time;
                cout << "FAST-LIO not ready" << endl;
                continue;
            }

            // flg_EKF_inited = !((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME);
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME
                             ? false
                             : true;

            /*** Build Voxel Map ***/
            if (flg_EKF_inited && !init_map) {
                pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(
                        new pcl::PointCloud<pcl::PointXYZI>);
                Eigen::Quaterniond q(state.rot_end);
                TransformLidar(state, p_imu, surf_feats_undistort, world_lidar);
                std::vector<pointWithCov> pv_list;
                for (size_t i = 0; i < world_lidar->size(); i++) {
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                            world_lidar->points[i].z;
                    V3D point_this(surf_feats_undistort->points[i].x,
                                   surf_feats_undistort->points[i].y,
                                   surf_feats_undistort->points[i].z);
                    // if z=0, error will occur in calcBodyCov. To be solved
                    if (point_this[2] == 0) {
                        point_this[2] = 0.001;
                    }
                    M3D cov;
                    calcBodyCov(point_this, ranging_cov, angle_cov, cov);

                    point_this += Lidar_offset_to_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    cov = state.rot_end * cov * state.rot_end.transpose() +
                          (-point_crossmat) * state.cov.block<3, 3>(0, 0) *
                          (-point_crossmat).transpose() +
                          state.cov.block<3, 3>(3, 3);
                    pv.cov = cov;
                    pv_list.push_back(pv);
                    Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                    sigma_pv[0] = sqrt(sigma_pv[0]);
                    sigma_pv[1] = sqrt(sigma_pv[1]);
                    sigma_pv[2] = sqrt(sigma_pv[2]);
                }

                BuildVoxelMap(pv_list,voxel_map);
                std::cout << "build voxel map" << std::endl;
                if (write_kitti_log) {
                    kitti_log(fp_kitti);
                }

                scanIdx++;
                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, voxel_map_pub);
                }
                init_map = true;
                cout << "Finish First Frame" << endl;
                continue;
            }


            /*** 4. VoxelGrid Downsample ***/
            auto t_downsample_start = std::chrono::high_resolution_clock::now();
            downSizeFilterSurf.setInputCloud(surf_feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            auto t_downsample_end = std::chrono::high_resolution_clock::now();
            std::cout << "feats size:" << surf_feats_undistort->size()
                      << ", down size:" << feats_down_body->size() << std::endl;
            auto t_downsample =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                            t_downsample_end - t_downsample_start)
                            .count() *
                    1000;

            sort(feats_down_body->points.begin(), feats_down_body->points.end(),
                 time_list);

            int rematch_num = 0;
            bool nearest_search_en = true;
            double total_residual;

            scan_match_time = 0.0;

            /*** 5. Calculate Body Cov ***/
            std::vector<M3D> body_var;
            std::vector<M3D> crossmat_list;
            auto calc_point_cov_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < feats_down_body->size(); i++) {
                V3D point_this(feats_down_body->points[i].x,
                               feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
                if (point_this[2] == 0) {
                    point_this[2] = 0.001;
                }
                M3D cov;
                if (calib_laser) {
                    calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
                } else {
                    calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list.push_back(point_crossmat);
                M3D rot_var = state.cov.block<3, 3>(0, 0);
                M3D t_var = state.cov.block<3, 3>(3, 3);
                body_var.push_back(cov);
            }

            auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
            double calc_point_cov_time =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                            calc_point_cov_end - calc_point_cov_start)
                            .count() *
                    1000;
            /*** 6. IESEKF Update ***/
            for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) {
                laserCloudOri->clear();
                laserCloudNoeffect->clear();
                total_residual = 0.0;

                std::vector<double> dist_list;
                std::vector<ptpl> ptpl_list;

                /** 6.1 Scan Matching based on Spatial Hashing **/
                vector<pointWithCov> pv_list;
                pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
                TransformLidar(state, p_imu, feats_down_body, world_lidar);
                for (size_t i = 0; i < feats_down_body->size(); i++) {
                    pointWithCov pv;
                    pv.point << feats_down_body->points[i].x,
                            feats_down_body->points[i].y, feats_down_body->points[i].z;
                    pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y,
                            world_lidar->points[i].z;
                    M3D cov = body_var[i];
                    M3D point_crossmat = crossmat_list[i];
                    M3D rot_var = state.cov.block<3, 3>(0, 0);
                    M3D t_var = state.cov.block<3, 3>(3, 3);
                    cov = state.rot_end * cov * state.rot_end.transpose() +
                          (-point_crossmat) * rot_var * (-point_crossmat.transpose()) +
                          t_var;
                    pv.cov = cov;
                    pv_list.push_back(pv);
                }

                auto scan_match_time_start = std::chrono::high_resolution_clock::now();
                std::vector<V3D> non_match_list;
                BuildResidualListOMP(voxel_map, pv_list, ptpl_list, non_match_list);
                auto scan_match_time_end = std::chrono::high_resolution_clock::now();

                effct_feat_num = ptpl_list.size();
                for (int i = 0; i < effct_feat_num; i++) {
                    total_residual += fabs(ptpl_list[i].dist);
                }
                res_mean_last = total_residual / effct_feat_num;
                scan_match_time +=
                        std::chrono::duration_cast<std::chrono::duration<double>>(
                                scan_match_time_end - scan_match_time_start)
                                .count() *
                        1000;

                /*** 6.2 Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                auto t_solve_start = std::chrono::high_resolution_clock::now();
                MatrixXd Hsub(effct_feat_num, 6);
                MatrixXd Hsub_T_R_inv(6, effct_feat_num);
                VectorXd R_inv(effct_feat_num);
                VectorXd meas_vec(effct_feat_num);

                /*** 6.2.1 Computation of Jacobian matrix H and Residual r ***/
                for (int i = 0; i < effct_feat_num; i++) {
                    V3D laser_p = ptpl_list[i].point;
                    V3D point_this(laser_p(0), laser_p(1), laser_p(2));
                    M3D cov;
                    if (calib_laser) {
                        calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
                    } else {
                        calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                    }

                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    V3D norm_p = ptpl_list[i].omega;
                    V3D norm_vec(norm_p(0), norm_p(1), norm_p(2));
                    V3D point_world = ptpl_list[i].point_world;
                    /*** Calculate the Covariance of Observation Noise ***/
                    Eigen::Matrix<double, 1, 3> J_abd;
                    Eigen::Matrix<double, 1, 3> J_pw;
                    V3D Omega = ptpl_list[i].omega;
                    double Omega_norm = ptpl_list[i].omega_norm;
                    double dist = ptpl_list[i].dist;
                    if (ptpl_list[i].main_direction == 0) { // ax+by+z+d = 0;
                        J_abd << point_world(0) * (1 - dist / (Omega_norm * Omega_norm)),
                                point_world(1) * (1 - dist / (Omega_norm * Omega_norm)), 1;
                    } else if (ptpl_list[i].main_direction == 1) { // ax+y+bz+d = 0;
                        J_abd << point_world(0) * (1 - dist / (Omega_norm * Omega_norm)),
                                point_world(2) * (1 - dist / (Omega_norm * Omega_norm)), 1;
                    } else { // x+ay+bz+d = 0;
                        J_abd << point_world(1) * (1 - dist / (Omega_norm * Omega_norm)),
                                point_world(2) * (1 - dist / (Omega_norm * Omega_norm)), 1;
                    }
                    J_abd /= Omega_norm;
                    J_pw = Omega.transpose() * state.rot_end / Omega_norm;
                    double sigma_l = J_abd * ptpl_list[i].plane_cov * J_abd.transpose();
                    R_inv(i) = 1.0 / (sigma_l + J_pw * cov * J_pw.transpose());
                    /*** Calculate the Measuremnt Jacobian matrix H ***/
                    V3D n = Omega / Omega_norm;
                    V3D A(point_crossmat * state.rot_end.transpose() * n);
                    Hsub.row(i) << VEC_FROM_ARRAY(A), n[0], n[1], n[2];
                    Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i),
                            A[2] * R_inv(i), n[0] * R_inv(i), n[1] * R_inv(i),
                            n[2] * R_inv(i);

                    /*** Measuremnt: distance to the closest surface ***/
                    // actually it is r_i = 0 - h(x)
                    meas_vec(i) = -dist;
                }
                MatrixXd K(DIM_STATE, effct_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;
                /*** 6.2.2 Iterative Kalman Filter Update ***/
                if (!flg_EKF_inited) {
                    cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
                    /*** only run in initialization period ***/
                    MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                    MatrixXd z_init(VD(9)::Zero());
                    H_init.block<3, 3>(0, 0) = M3D::Identity();
                    H_init.block<3, 3>(3, 3) = M3D::Identity();
                    H_init.block<3, 3>(6, 15) = M3D::Identity();
                    z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
                    z_init.block<3, 1>(0, 0) = -state.pos_end;

                    auto H_init_T = H_init.transpose();
                    auto &&K_init =
                            state.cov * H_init_T *
                            (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity())
                                    .inverse();
                    solution = K_init * z_init;

                    state.resetpose();
                    EKF_stop_flg = true;
                } else {
                    auto &&Hsub_T = Hsub.transpose();
                    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
                    MD(DIM_STATE, DIM_STATE) &&K_1 =
                            (H_T_H + (state.cov).inverse()).inverse();
                    K = K_1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv;
                    auto vec = state_propagat - state;
                    solution = K * meas_vec + vec - K * Hsub * vec.block<6, 1>(0, 0);

                    int minRow, minCol;
                    state += solution;
                    rot_add = solution.block<3, 1>(0, 0);
                    t_add = solution.block<3, 1>(3, 0);

                    if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) {
                        flg_EKF_converged = true;
                    }

                    deltaR = rot_add.norm() * 57.3;
                    deltaT = t_add.norm() * 100;
                }
                euler_cur = RotMtoEuler(state.rot_end);
                /*** Rematch Judgement ***/
                nearest_search_en = false;
                if (flg_EKF_converged ||
                    ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
                    nearest_search_en = true;
                    rematch_num++;
                }
                /*** 6.2.3 Convergence Judgements and Covariance Update ***/
                if (!EKF_stop_flg &&
                    (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
                    if (flg_EKF_inited) {
                        /*** Covariance Update ***/
                        G.setZero();
                        G.block<DIM_STATE, 6>(0, 0) = K * Hsub;
                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;

                        geoQuat = tf::createQuaternionMsgFromRollPitchYaw(
                                euler_cur(0), euler_cur(1), euler_cur(2));

                        VD(DIM_STATE) K_sum = K.rowwise().sum();
                        VD(DIM_STATE) P_diag = state.cov.diagonal();
                    }
                    EKF_stop_flg = true;
                    for (int i = 0; i < effct_feat_num; i++) {
                        PointType pi_body;
                        pi_body.x = ptpl_list[i].point(0);
                        pi_body.y = ptpl_list[i].point(1);
                        pi_body.z = ptpl_list[i].point(2);
                        laserCloudOri->push_back(pi_body);
                    }
                }
                auto t_solve_end = std::chrono::high_resolution_clock::now();
                solve_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                        t_solve_end - t_solve_start)
                                      .count() *
                              1000;
                if (EKF_stop_flg)
                    break;

            }
            /*** 7. Update Voxel Map ***/
            auto map_incremental_start = std::chrono::high_resolution_clock::now();
            pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(
                    new pcl::PointCloud<pcl::PointXYZI>);
            TransformLidar(state, p_imu, feats_down_body, world_lidar);
            std::vector<pointWithCov> pv_list;
            for (size_t i = 0; i < world_lidar->size(); i++) {
                pointWithCov pv;
                pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                        world_lidar->points[i].z;
                M3D point_crossmat = crossmat_list[i];
                M3D cov = body_var[i];
                cov = state.rot_end * cov * state.rot_end.transpose() +
                      (-point_crossmat) * state.cov.block<3, 3>(0, 0) *
                      (-point_crossmat).transpose() +
                      state.cov.block<3, 3>(3, 3);
                pv.cov = cov;
                pv_list.push_back(pv);
            }
            std::sort(pv_list.begin(), pv_list.end(), var_contrast);
            UpdateVoxelMap(pv_list,voxel_map);
            auto map_incremental_end = std::chrono::high_resolution_clock::now();
            map_incremental_time =
                    std::chrono::duration_cast<std::chrono::duration<double>>(
                            map_incremental_end - map_incremental_start)
                            .count() *
                    1000;


            total_time = t_downsample + scan_match_time + solve_time +
                         map_incremental_time + undistort_time + calc_point_cov_time;

            if (enable_write && scanIdx % 10 == 0) {
                foutC << std::fixed << std::setprecision(10) << Measures.lidar_beg_time << " ";
                foutC << state.pos_end[0] << " " << state.pos_end[1] << " " << state.pos_end[2] << " ";

                foutC << total_time << "  ";
                foutC << 0 << " " << 0 << " " << 0 << " " << 0 << endl;
            }
            /*** 8. Publish functions:  ***/
            publish_odometry(pubOdomAftMapped);
            publish_path(pubPath);
            tf::Transform transform;
            tf::Quaternion q;
            transform.setOrigin(
                    tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));
            q.setW(geoQuat.w);
            q.setX(geoQuat.x);
            q.setY(geoQuat.y);
            q.setZ(geoQuat.z);
            transform.setRotation(q);
            TransformLidar(state, p_imu, feats_down_body, world_lidar);
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(*world_lidar, pub_cloud);
            pub_cloud.header.stamp =
                    ros::Time::now(); //.fromSec(last_timestamp_lidar);
            pub_cloud.header.frame_id = "camera_init";
            if (publish_point_cloud) {
                publish_surf_frame_world(pubLaserCloudSurfFull, pub_point_cloud_skip);
            }

            if (scanIdx % 200 == 0) {
                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, voxel_map_pub);
                }
            }

            publish_effect(pubLaserCloudEffect);

            frame_num++;
            mean_raw_points = mean_raw_points * (frame_num - 1) / frame_num +
                              (double) (surf_feats_undistort->size()) / frame_num;
            mean_ds_points = mean_ds_points * (frame_num - 1) / frame_num +
                             (double) (feats_down_body->size()) / frame_num;
            mean_effect_points = mean_effect_points * (frame_num - 1) / frame_num +
                                 (double) effct_feat_num / frame_num;

            undistort_time_mean = undistort_time_mean * (frame_num - 1) / frame_num +
                                  (undistort_time) / frame_num;
            down_sample_time_mean =
                    down_sample_time_mean * (frame_num - 1) / frame_num +
                    (t_downsample) / frame_num;
            calc_cov_time_mean = calc_cov_time_mean * (frame_num - 1) / frame_num +
                                 (calc_point_cov_time) / frame_num;
            scan_match_time_mean =
                    scan_match_time_mean * (frame_num - 1) / frame_num +
                    (scan_match_time) / frame_num;
            ekf_solve_time_mean = ekf_solve_time_mean * (frame_num - 1) / frame_num +
                                  (solve_time) / frame_num;
            map_update_time_mean =
                    map_update_time_mean * (frame_num - 1) / frame_num +
                    (map_incremental_time) / frame_num;

            aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num +
                              (total_time) / frame_num;

            time_log_counter++;
            cout << "pos:" << state.pos_end.transpose() << endl;
            cout << "[ Time ]: "
                 << "average undistort: " << undistort_time_mean << std::endl;
            cout << "[ Time ]: "
                 << "average down sample: " << down_sample_time_mean << std::endl;
            cout << "[ Time ]: "
                 << "average calc cov: " << calc_cov_time_mean << std::endl;
            cout << "[ Time ]: "
                 << "average scan match: " << scan_match_time_mean << std::endl;
            cout << "[ Time ]: "
                 << "average solve: " << ekf_solve_time_mean << std::endl;
            cout << "[ Time ]: "
                 << "average map incremental: " << map_update_time_mean << std::endl;
            cout << "[ Time ]: "
                 << " average total " << aver_time_consu << endl;
            cout << "--------------------------------------------" << endl;
            if (write_kitti_log) {
                kitti_log(fp_kitti);
            }

            scanIdx++;
        }
        status = ros::ok();
        rate.sleep();
    }
    return 0;
}
