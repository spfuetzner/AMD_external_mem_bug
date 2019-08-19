#version 460 

layout(points,invocations=3) in;
layout(triangle_strip,max_vertices = 4) out;

layout (location = 0) in flat uint node_id[1];
layout (location = 1) in vec4 center[1];
layout (location = 2) in vec4 radius[1];

layout (location = 0) out flat uint node_id_out;


layout (push_constant) uniform RenderData
{
	mat4 proj_mat;
	vec4 cam_pos;
	uint num_nodes;
};

void main()
{
	// This node id seems to be wrong when read out in fragment shader
	// with all VK1_1 beta and main release driver since 39X.XX last working 389.20 (see bbox.frag)


	vec3 face_normal = vec3(0);
	vec3 edge_base_0 = vec3(0);
	vec3 edge_base_1 = vec3(0);

	int id = gl_InvocationID;

	switch (id)
	{
	case 0:
		face_normal.x = radius[0].x;
		edge_base_0.y = radius[0].y;
		edge_base_1.z = radius[0].z;
		break;
	case 1:	
		face_normal.y = radius[0].y;
		edge_base_1.x = radius[0].x;
		edge_base_0.z = radius[0].z;
		break;
	case 2:
		face_normal.z = radius[0].z;
		edge_base_0.x = radius[0].x;
		edge_base_1.y = radius[0].y;
		break;
	}
	vec3 world_center = center[0].xyz;

	vec3 world_normal = face_normal;
	vec3 world_pos = world_center + world_normal;
	float projection = sign(dot(world_pos - cam_pos.xyz,world_normal));

	projection *= -1;

	face_normal *= projection;
	edge_base_1 *= projection;

	node_id_out = node_id[0];
	gl_Position = proj_mat * vec4(world_center + (face_normal - edge_base_0 - edge_base_1),1);
	EmitVertex();		
	
	node_id_out = node_id[0];
	gl_Position = proj_mat  * vec4(world_center + (face_normal + edge_base_0 - edge_base_1),1);
	EmitVertex();	
	
	node_id_out = node_id[0];
	gl_Position = proj_mat  * vec4(world_center +  (face_normal - edge_base_0 + edge_base_1),1);
	EmitVertex();		
	
	node_id_out = node_id[0];
	gl_Position = proj_mat  * vec4(world_center +  (face_normal + edge_base_0 + edge_base_1),1);
	EmitVertex();

	EndPrimitive();		
} 