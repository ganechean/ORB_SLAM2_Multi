/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/io/pcd_io.h>
#include <valarray>
#include "Converter.h"

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = (resolution_>0)?resolution_:resolution;
    cout << endl << "PointCloudMapping.Resolution: " << resolution << endl;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    
    //viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    /****************************************************/



    /****************************************************/
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    PointCloudMapping::test();
    //viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{

    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    //cout<<kf->GetPose()<<endl;

    unique_lock<mutex> lck(keyframeMutex);
//*
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>10)
                continue;
            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    //cout<<kf->GetPose()<<endl;
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    return cloud;
}


void PointCloudMapping::test()
{
    for(size_t i=0;i<keyframes.size();i++)// save the optimized pointcloud
    {
        //cout<<"keyframe "<<i<<" ..."<<endl;
        PointCloud::Ptr tp = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( tp );
        voxel.filter( *tmp );
        *globalMap += *tmp;
    }
    //PointCloud::Ptr tmp(new PointCloud());
    //sor.setInputCloud(globalMap);
    //sor.filter(*tmp);
    //globalMap->swap( *tmp );
    //pcl::io::savePCDFileBinary ( "optimized_pointcloud.pcd", *globalMap );
    //cout<<"Save point cloud file successfully!"<<endl;
    /***********************************************************/
    //pcl::PointCloud<pcl::PointXYZRGBA> cloudt;
    //pcl::io::loadPCDFile<pcl::PointXYZRGBA> ( "optimized_pointcloud.pcd", cloudt );
    cout << "copy data into octomap..." << endl;


    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGBA> ()) ;



    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZRGBA> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setDistanceThreshold(0.01);
    seg.setMaxIterations (2000);
    seg.setInputCloud(globalMap);
    seg.segment(*inliers, *coefficients);

    cout << "Model coefficients: " << coefficients->values[0] << " "
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " "
                                      << coefficients->values[3] << endl;

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f transform2 = Eigen::Matrix4f::Identity();

    cout << atan(coefficients->values[0]/coefficients->values[2]) << ", " << atan(coefficients->values[1]/coefficients->values[2]) << endl;

    double xtoz = atan(coefficients->values[0]/coefficients->values[2]);
    double ytoz = atan(coefficients->values[1]/coefficients->values[2]);

    transform(0,0) = 0		;transform(0,1) = -1		;transform(0,2) = 0;
    transform(1,0) = 1		;transform(1,1) = 0		;transform(1,2) = 0;
    transform(2,0) = 0		;transform(2,1) = 0		;transform(2,2) = 1;

    transform2(0,0) = 1	;transform2(0,1) = 0		;transform2(0,2) = 0;
    transform2(1,0) = 0	;transform2(1,1) = cos(ytoz)	;transform2(1,2) = -sin(ytoz);
    transform2(2,0) = 0	;transform2(2,1) = sin(ytoz)	;transform2(2,2) = cos(ytoz);

    pcl::transformPointCloud (*globalMap, *transformed_cloud, transform*transform2);
/*
    pcl::visualization::CloudViewer viewer("lhc_viewer");

    viewer.showCloud(globalMap);
    sleep(10);

    viewer.showCloud(transformed_cloud);
    sleep(10);

    pcl::transformPointCloud (*globalMap, *transformed_cloud, transform2);
    viewer.showCloud(transformed_cloud);
    sleep(10);

    pcl::transformPointCloud (*globalMap, *transformed_cloud, transform*transform2);
    viewer.showCloud(transformed_cloud);
    while (!viewer.wasStopped ());

*/
    // Create an octree object，set resolution as 0.03.
    octomap::OcTree tree( 0.005 );
    for (auto p:transformed_cloud->points)
    {
        // Insert the point of Point Cloud to octomap
        tree.updateNode( octomap::point3d(p.x, p.y, p.z), true );
    }

    // update octomap
    tree.updateInnerOccupancy();
    // save octomap
    tree.writeBinary( "global_octomap.bt" );
    cout<<"save octomap ... done."<<endl;

}


