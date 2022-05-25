#ifndef FOURCAM_PROCESSOR_H_
#define FOURCAM_PROCESSOR_H_

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
// #include <message_filters/sync_policies/approximate_time.h>
#include <image_transport/subscriber_filter.h>

#include <basalt/optical_flow/optical_flow.h>
#include <basalt/calibration/calibration.hpp>


#include <tbb/tbb.h>
#include <stdlib.h>
#include <basalt/utils/vio_config.h>
#include <time.h>
/**
 * This is an abstract base class for stereo image processing nodes.
 * It handles synchronization of input topics (approximate or exact)
 * and checks for sync errors.
 * To use this class, subclass it and implement the imageCallback() method.
 */



class FourCamProcessor
{

public:
	struct Parameters {
		int queue_size;

		std::string cam0_topic, cam1_topic, cam2_topic, cam3_topic;
		std::string cam0_info_topic, cam1_info_topic, cam2_info_topic, cam3_info_topic;
	};

	// static constexpr size_t N = 2;

	// struct SyncedData {
	// 	ros::Time timestamp;
	// 	std::array<sensor_msgs::Image::ConstPtr, N> image_ptrs;
	// 	std::array<sensor_msgs::CameraInfo::ConstPtr, N> info_ptrs;
	// };

	// tbb::concurrent_bounded_queue<SyncedData> multi_camera_buffer;
	basalt::OpticalFlowInput::Ptr last_img_data;
  	tbb::concurrent_bounded_queue<basalt::OpticalFlowInput::Ptr>* image_data_queue = nullptr;

private:

	uint64_t get_monotonic_now(void)
	{
		struct timespec spec;
		clock_gettime(CLOCK_MONOTONIC, &spec);

		return spec.tv_sec * 1000000000ULL + spec.tv_nsec;
	}

	basalt::VioConfig config;
	Parameters param;

	// subscriber
	image_transport::SubscriberFilter cam0_sub_, cam1_sub_, cam2_sub_, cam3_sub_;
	// message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_, right_info_sub_;
	

	// for sync checking
	ros::WallTimer check_synced_timer_;
	int cam0_received_, cam1_received_, cam2_received_, cam3_received_, all_received_;

	typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::Image> ExactPolicy;
	typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
	ExactSync exact_sync_;

	// for sync checking
	static void increment(int* value){ ++(*value); }

	// // this callback is only called when all messages are synced at one time
	// void dataCb(	const sensor_msgs::ImageConstPtr& l_image_msg,
	// 				const sensor_msgs::ImageConstPtr& r_image_msg,
	// 				const sensor_msgs::CameraInfoConstPtr& l_info_msg,
	// 				const sensor_msgs::CameraInfoConstPtr& r_info_msg){	
	// 	//int _seq = l_image_msg->header.seq;
	// 	all_received_++;

	// 	// imageCallback(l_image_msg, r_image_msg, l_info_msg, r_info_msg);

	// 	SyncedData data;

	// 	data.timestamp = l_image_msg->header.stamp;
	// 	data.image_ptrs[0] = l_image_msg;
	// 	data.image_ptrs[1] = r_image_msg;
	// 	data.info_ptrs[0] = l_info_msg;
	// 	data.info_ptrs[1] = r_info_msg;

	// 	multi_camera_buffer.push(data);
	// }

	void basaltDataCb(	const sensor_msgs::ImageConstPtr& cam0_image_msg,
					const sensor_msgs::ImageConstPtr& cam1_image_msg,
					const sensor_msgs::ImageConstPtr& cam2_image_msg,
					const sensor_msgs::ImageConstPtr& cam3_image_msg){	

		all_received_++;
		basalt::OpticalFlowInput::Ptr data(new basalt::OpticalFlowInput);
      	data->img_data.resize(4); //NUM_CAMS = 4
		sensor_msgs::ImageConstPtr img_msg;
		for (int i = 0; i < 4; i++) {

			switch (i)
			{
				case 0:
					img_msg = cam0_image_msg;
					break;
				case 1:
					img_msg = cam1_image_msg;
					break;
				case 2:
					img_msg = cam2_image_msg;
					break;
				case 3:
					img_msg = cam3_image_msg;
					break;
			}
			
			data->t_ns = img_msg->header.stamp.toNSec();

			ROS_INFO_STREAM_THROTTLE(5.0, "stereo image callback delay is " << ( get_monotonic_now() - data->t_ns ) / 1e6 << " ms");
			if (!img_msg->header.frame_id.empty() &&
				std::isdigit(img_msg->header.frame_id[0])) {
				data->img_data[i].exposure = std::stol(img_msg->header.frame_id) * 1e-9;
			} 
			else {
				data->img_data[i].exposure = -1;
			}

			data->img_data[i].img.reset(new basalt::ManagedImage<uint16_t>(
				img_msg->width, img_msg->height));

			if (img_msg->encoding == "mono8") {
				const uint8_t *data_in = img_msg->data.data();
				uint16_t *data_out = data->img_data[i].img->ptr;

				for (size_t i = 0; i < img_msg->data.size(); i++) {
					int val = data_in[i];
					val = val << 8;
					data_out[i] = val;
				}

			} else if (img_msg->encoding == "mono16") {
				std::memcpy(data->img_data[i].img->ptr, img_msg->data.data(), img_msg->data.size());
			} else if (img_msg->encoding == "bgra8") {
				const uint8_t *data_in = img_msg->data.data();
				uint16_t *data_out = data->img_data[i].img->ptr;

				for (size_t i = 0; i < img_msg->data.size()/4; i++) {
					size_t j = i * 4;
					int val = (data_in[j] + data_in[j+1] + data_in[j+2])/3;
					val = val << 8;
					data_out[i] = val;
				}

			} else {
				std::cerr << "Encoding " << img_msg->encoding << " is not supported."
							<< std::endl;
				std::abort();
			}
		}

		last_img_data = data;
		if (image_data_queue) {
			if(image_data_queue->try_push(data)){
				if(config.vio_debug)
					std::cout<< "**current image queue size is: "<<image_data_queue->size()<<std::endl<<std::endl;
			}
			else{
      			std::cout<<"image data buffer is full: "<<image_data_queue->size()<<std::endl;
      			abort();
    			}
			}
	}

	void checkInputsSynchronized()
	{
		if (!all_received_)
			return;
		int threshold = 3 * all_received_;
		if (cam0_received_ >= threshold || cam1_received_ >= threshold || 
				cam2_received_ >= threshold || cam3_received_ >= threshold) {
			ROS_WARN("[stereo_processor] Low number of synchronized cam0/cam1/cam2/cam3 tuples received.\n"
								"Cam0 images received:       %d (topic '%s')\n"
								"Cam1 images received:      %d (topic '%s')\n"
								"Cam2 images received:  %d (topic '%s')\n"
								"Cam3 images info received: %d (topic '%s')\n"
								"Synchronized tuples: %d\n"
								"Possible issues:\n"
								"\t* stereo_image_proc is not running.\n"
								"\t  Does `rosnode info %s` show any connections?\n"
								"\t* The cameras are not synchronized.\n"
								"\t  Try restarting the node with parameter _approximate_sync:=True\n"
								"\t* The network is too slow. One or more images are dropped from each tuple.\n"
								"\t  Try restarting the node, increasing parameter 'queue_size' (currently %d)",
								cam0_received_, cam0_sub_.getTopic().c_str(),
								cam1_received_, cam1_sub_.getTopic().c_str(),
								cam2_received_, cam2_sub_.getTopic().c_str(),
								cam3_received_, cam3_sub_.getTopic().c_str(),
								all_received_, ros::this_node::getName().c_str(), param.queue_size);
		}
	}


public:
	
	/**
	 * Constructor, subscribes to input topics using image transport and registers
	 * callbacks.
	 * \param transport The image transport to use
	 */

	// FourCamProcessor(int queue_size, const std::string& left_topic, const std::string& right_topic, 
	// 				const std::string& left_info_topic, const std::string& right_info_topic ) : FourCamProcessor( Parameters{queue_size,left_topic,right_topic, left_info_topic, right_info_topic})
	// {
	// }

	FourCamProcessor(const basalt::VioConfig& config, const Parameters& param) :  config(config), param(param), 
		cam0_received_(0), cam1_received_(0), cam2_received_(0), cam3_received_(0), all_received_(0), 
		exact_sync_(ExactPolicy(param.queue_size), cam0_sub_, cam1_sub_, cam2_sub_, cam3_sub_)
	{
		// Read local parameters
		ros::NodeHandle local_nh("~");

		// Resolve topic names
		ros::NodeHandle nh;

		// Subscribe to four input topics.
		ROS_INFO("4cam processor: Subscribing to:\n\t* %s\n\t* %s\n\t* %s\n\t* %s", 
				param.cam0_topic.c_str(), param.cam1_topic.c_str(),
				param.cam2_topic.c_str(), param.cam3_topic.c_str());
		image_transport::ImageTransport it(nh);
		//image_transport::TransportHints hints(transport,ros::TransportHints().tcpNoDelay());
		cam0_sub_.subscribe(it, param.cam0_topic, 10); //, hints); // http://docs.ros.org/diamondback/api/image_transport/html/classimage__transport_1_1TransportHints.html
		cam1_sub_.subscribe(it, param.cam1_topic, 10); //, hints);
		cam2_sub_.subscribe(it, param.cam2_topic, 10); //, ros::TransportHints().tcpNoDelay());
		cam3_sub_.subscribe(it, param.cam3_topic, 10); //,  ros::TransportHints().tcpNoDelay());

		// Complain every 15s if the topics appear unsynchronized
		cam0_sub_.registerCallback(boost::bind(FourCamProcessor::increment, &cam0_received_));
		cam1_sub_.registerCallback(boost::bind(FourCamProcessor::increment, &cam1_received_));
		cam2_sub_.registerCallback(boost::bind(FourCamProcessor::increment, &cam2_received_));
		cam3_sub_.registerCallback(boost::bind(FourCamProcessor::increment, &cam3_received_));
		check_synced_timer_ = nh.createWallTimer(ros::WallDuration(15.0),
				boost::bind(&FourCamProcessor::checkInputsSynchronized, this));

		// Synchronize input topics.
		exact_sync_.registerCallback(boost::bind(&FourCamProcessor::basaltDataCb, this, _1, _2, _3, _4));
	}

	/**
	 * Implement this method in sub-classes 
	 */
	// virtual void imageCallback(	const sensor_msgs::ImageConstPtr l_image_msg,
	// 							const sensor_msgs::ImageConstPtr r_image_msg,
	// 							const sensor_msgs::CameraInfoConstPtr l_info_msg,
	// 							const sensor_msgs::CameraInfoConstPtr r_info_msg) = 0;

};

#endif

