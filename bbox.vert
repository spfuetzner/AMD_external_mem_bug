#version 460

layout (location = 0) in uint point_id;
layout (location = 0) out uint node_id;

layout (location = 1) out vec4 center;
layout (location = 2) out vec4 radius;

layout (push_constant) uniform RenderData
{
	mat4 proj_mat;
	vec4 cam_pos;
	uint num_nodes;
};

struct AABB
{
	vec4 center;
	vec4 radius;
};

layout (binding = 0) buffer AABBStorage
{
	AABB aabb[];
};

void main()
{
	node_id = point_id; // it is also valid to use gl_VertexIndex instead
	center = aabb[gl_VertexIndex].center;
	radius = aabb[gl_VertexIndex].radius;
}