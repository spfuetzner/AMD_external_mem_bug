#version 460

layout (location = 0) in flat uint node_id;
layout (location = 0) out vec3 color;

layout (push_constant) uniform RenderData
{
	mat4 proj_mat;
	vec4 cam_pos;
	uint num_val_nodes;
};

void main()
{
	
	// if node id is invalid then set red color to the created bounding boxes
	if (node_id >= num_val_nodes) // remove the if and you ll get crash with official driver releases since 397.31 and before 397.64
	{	
		color = vec3(1,0,0);
	}
	else
	{
		color = vec3(1,1,1);
	}
}