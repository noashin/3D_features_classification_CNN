#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/fpfh.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

//Struct to read pcd files from http://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/
struct PointXYZRGBIM
{
  union
  {
    struct
    {
      float x;
      float y;
      float z;
      float rgb;
      float imX;
      float imY;
    };
    float data[6];
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZRGBIM,
                                    (float, x, x)
                                    (float, y, y)
                                    (float, z, z)
                                    (float, rgb, rgb)
                                    (float, imX, imX)
                                    (float, imY, imY)
)

//variables used to keep track over how many files were converted and what is the total size.
// If total_size > 200 Giga stop the program.
static int files_converted = 0;
static uintmax_t total_size = 0;
const uintmax_t MAX_SIZE = 200000000000;

/*
This function iterates through all pcd files in directory 
*/
void iterate_over_folder(const std::string& source_path, const std::string& target_path)
{
    boost::filesystem::path target_dir(target_path);
    boost::filesystem::path source_dir(source_path);
    boost::filesystem::directory_iterator it(source_dir), eod;

    BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(it, eod))   
    { 
        if(is_regular_file(file_path))
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<PointXYZRGBIM>::Ptr cloud_im (new pcl::PointCloud<PointXYZRGBIM>);
            if (pcl::io::loadPCDFile<pcl::PointXYZ> (file_path.string(), *cloud) == -1 ||
            pcl::io::loadPCDFile<PointXYZRGBIM> (file_path.string(), *cloud_im) == -1) //* load the file
            {
                PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
            }

            
            //Estimate normals
            pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal> ());
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree0(new pcl::search::KdTree<pcl::PointXYZ>);

            tree0->setInputCloud(cloud);
            n.setInputCloud(cloud);
            n.setSearchMethod(tree0);
            n.setKSearch(20);
            n.compute(*normals);

            // Create the FPFH estimation class, and pass the input dataset+normals to it
            pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
            fpfh.setInputCloud (cloud);
            fpfh.setInputNormals (normals);

            // Create an empty kdtree representation, and pass it to the FPFH estimation object.
            // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

            fpfh.setSearchMethod (tree);

            // Output datasets
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

            // Use all neighbors in a sphere of radius 5cm
            // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
            fpfh.setRadiusSearch (0.05);

            // Compute the features
            fpfh.compute (*fpfhs);

            //Save features
            std::string fpfh_file_name =  file_path.stem().string() + "_fpfh.csv";
            boost::filesystem::path fpfh_folder_path = target_dir / source_dir.stem();
            boost::filesystem::create_directory(fpfh_folder_path);
            boost::filesystem::path fpfh_file_path = fpfh_folder_path / fpfh_file_name;
            const char * fpfh_file_path_c = fpfh_file_path.string().c_str();

            std::ofstream myfile;
            myfile.open(fpfh_file_path_c);
            
            std::cout << "Loaded "
                << cloud->width * cloud->height
                << " data points from test_pcd.pcd with the following fields: "
                << std::endl;


            if (myfile.is_open())
            {
                for (int i = 0; i < fpfhs->points.size(); ++i){
                    
                    uint32_t rgb = *reinterpret_cast<int*>(&cloud_im->points[i].rgb);
                    uint8_t r = (rgb >> 16) & 0x0000ff;
                    uint8_t g = (rgb >> 8)  & 0x0000ff;
                    uint8_t b = (rgb)       & 0x0000ff;
                    
                    for (int j = 0; j < 33; ++j){
                        myfile << fpfhs->points[i].histogram[j] << " " ;
                    }

                    myfile << (int)r << " " << (int)g << " " << (int)b << " " << 
                        cloud_im->points[i].imX << " " << cloud_im->points[i].imY << " \n";
                }
                myfile.close();
                std::cout << "saved file: " << fpfh_file_path.string() << "\n";
                files_converted ++;
                total_size += boost::filesystem::file_size(fpfh_file_path);
                if (total_size >= MAX_SIZE)
                {
                    std::cout << "program converted " << files_converted << " files and exceeded maximal size set to: "
                    << MAX_SIZE << " bytes." << std::endl;
                }
                else 
                {
                    std::cout << files_converted << std::endl;
                }
            }
            else std::cout << "Unable to open file";
        } 
    }

}


int main()
{
    const char * homedir = getenv("HOME");
    std::string objects[] = {"coffee_mug", "greens", "soda_can", "ball", "comb", "hand_towel", "sponge", "dry_battery", "instant_noodles", "peach", "stapler", "flashlight",
"keyboard", "binder", "food_bag", "kleenex", "pitcher", "toothbrush", "bowl", "food_box", "plate", "toothpaste", "calculator", "food_can", "lightbulb", "pliers", "water_bottle"
"camera", "food_cup", "cap", "food_jar", "marker", "rubber_eraser", "cell_phone", "garlic", "scissors", "cereal_box", "glue_stick", "notebook" };
    
    std::string datasets_dir_string = std::string(homedir) + "/datasets/rgbd-dataset/";
    std::string target_dir = "/mnt/raid/dnn/fpfh";
    
    //std::string datasets_dir_string = "/Users/smartMac/Downloads/rgbd-dataset";
    //std::string target_dir = "/Users/smartMac/lab_rot_1/pcl_proj/fpfh";
    

    boost::filesystem::path datasets_dir(datasets_dir_string);
    boost::filesystem::directory_iterator it(datasets_dir_string), eod;

    for(int i = 0; i < 13; ++i)
    {

        boost::filesystem::path object_path(datasets_dir_string + objects[i]);
        if(boost::filesystem::is_directory(object_path))
        {
            boost::filesystem::directory_iterator it2(object_path), eod2;
            BOOST_FOREACH(boost::filesystem::path const &instance_path, std::make_pair(it2, eod2))
            {
                if(boost::filesystem::is_directory(instance_path))
                {
                    iterate_over_folder(instance_path.string(), target_dir);
                }
            }
        }
        else
        {
            std::cout << object_path.string() << " not a directory" << std::endl;
        }

    }  
    return 0;
}