#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "main.h"
#include <vector>
#include <map>
#include <tuple>
#include <Eigen/Dense>
#include <Eigen/Core>

#include "PhysicsEngine.h"
#include "SimObject.h"
#include "Robot.h"
#include "Edge.h"
#include "Snapshot.h"

using namespace std;
using namespace Eigen;

class Environment
{
private:
	
	//PHYSICS
	PhysicsEngine physics_handler;

	//POINTS
	int num_points;
	Matrix <double, 2, Dynamic> points_pos;
	Matrix <double, 2, Dynamic> points_pos_last;
	Matrix <double, 2, Dynamic> points_vel;
	Matrix <double, 2, Dynamic> points_vel_true;
	Matrix <double, 2, Dynamic> points_mass;
	Matrix <bool, 2, Dynamic> points_fixed;

	//EDGES
	vector <Edge> edges;
	Matrix <int, 1, Dynamic> a_index;
	Matrix <int, 1, Dynamic> b_index;
	Matrix <double, 1, Dynamic> length_eq;
	Matrix <double, 1, Dynamic> length_eq_goal;
	Matrix <double, 1, Dynamic> init_length_eq;
	Matrix <double, 1, Dynamic> spring_const;


	//vector <Vector2d_old> points_pos_old;
	//vector <Vector2d_old> points_vel_old;
	//vector <double> points_mass_old;
	//vector <bool> points_fixed_old;


	//OBJECTS
	vector <SimObject*> objects;
	map <string, int> object_name_to_index;

	//HISTORY
	map <long int, Snapshot> history;

public:
	Environment();
	~Environment();

	vector <bool> point_is_colliding;

	void init();

	//int createPoint(Vector2d_old pos, double mass);
	//int create_edge(Edge edge);

	void create_points(vector <Vector2d>* pos, vector <Vector2d>* vel, vector <double>* mass, vector <bool>* fixed);
	void create_edges(vector <Edge>* new_edges);
	void swap_edge(int edge_index);
	void init_robot(string robot_name);
	bool add_object_name(string object_name, int index);
	void set_surface_edge_color(int color);

	bool step();
	bool special_step();
	void set_robot_action(string robot_name, Ref <Matrix <double, Dynamic, 2>> action);
	void save_snapshot(long int sim_time);
	bool revert_to_snapshot(long int sim_time);

	Matrix <double, 2, Dynamic>* get_pos();
	Matrix <double, 2, Dynamic>* get_vel();
	Matrix <double, 2, Dynamic>* get_mass();
	Matrix <bool, 2, Dynamic>* get_fixed();

	int get_num_points();
	int get_num_edges();

	vector<Edge>* get_edges();
	vector<SimObject*>* get_objects();
	Robot* get_robot(string robot_name);

	Ref <MatrixXd> get_pos_at_time(long int sim_time);
	Ref <MatrixXd> get_vel_at_time(long int sim_time);

	Ref <MatrixXd> object_pos_at_time(long int sim_time, string object_name);
	Ref <MatrixXd> object_pos_at_time_matrix_for_robot(long int sim_time, string object_name);
	Ref <MatrixXd> object_pos_at_time_matrix_for_robot_2(long int sim_time, string object_name);
	Ref <MatrixXd> object_vel_at_time(long int sim_time, string object_name);
	double object_orientation_at_time(long int sim_time, string object_name);
	void translate_object(double x, double y, string object_name);

	void print_poses();


	// 这个变量存的是下标到网格二维坐标的映射，用来解析obs_env的位置信息
	vector<pair<int, int>> idx_2_xy;

	// 点阵大小
	int robot_grid_width;
	int robot_grid_height;


	//每一个方块，对应4个角上的点，的xy
	map<pair<int, int>, vector<int>> idx_2_xy_block;

	// 块阵大小
	int robot_grid_width_block;
	int robot_grid_height_block;

	Matrix <double, 2, Dynamic> ret;

	//vector<Vector2d_old>* getPointsPos();
	//vector<bool>* getPointsFixed();
	//vector<double>* getMasses();
};

#endif // !ENVIRONMENT_H
