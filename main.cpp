/*
	The example is very simplified and does not do what described there besides creating bounding boxes in the geom shader and shows them.
	Additionally it creates external image memory line: 614 and uses it to import to opengl line: 1112
	The so created opengl textures are attached to fbo. The FBO is used to blit attachment content to opengl default swap chain (default fbo)

	Somehow are the rendering result completely broken.

	Thanks in advance!
*/
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#pragma comment(lib,"opengl32.lib")
#include <gl/glew.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#include <glfw/glfw3.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/fwd.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <sstream>

#include "bbox.vert.h"
#include "bbox.geom.h"
#include "bbox.frag.h"

////////////////////////////////////////////////////////////////////////////////
// some utility code
///////////////////////////////////////////////////////////////////////////////
bool isInstanceExtensionAvailable(std::string const& ext)
{
	std::vector<std::string> avail_exts;
	for (auto const& e : vk::enumerateInstanceExtensionProperties())
		avail_exts.push_back(e.extensionName);
	return std::find(avail_exts.begin(), avail_exts.end(), ext) != avail_exts.end();
}

bool isInstanceLayerAvailable(std::string const& layer)
{
	std::vector<std::string> avail_layers;
	for (auto const& l : vk::enumerateInstanceLayerProperties())
		avail_layers.push_back(l.layerName);
	return std::find(avail_layers.begin(), avail_layers.end(), layer) != avail_layers.end();
}

const char* OGLErrorToString(GLenum error)
{
	switch (error)
	{
	case GL_NO_ERROR:
		return "GL_NO_ERROR: No error has been recorded.";
	case GL_INVALID_ENUM:
		return "GL_INVALID_ENUM: An unacceptable value is specified for an enumerated argument. The offending function is ignored, having no side effect other than to set the error flag.";
	case GL_INVALID_VALUE:
		return "GL_INVALID_VALUE: A numeric argument is out of range. The offending function is ignored, having no side effect other than to set the error flag.";
	case GL_INVALID_OPERATION:
		return "GL_INVALID_OPERATION: The specified operation is not allowed in the current state. The offending function is ignored, having no side effect other than to set the error flag.";
	case GL_STACK_OVERFLOW:
		return "GL_STACK_OVERFLOW: This function would cause a stack overflow. The offending function is ignored, having no side effect other than to set the error flag.";
	case GL_STACK_UNDERFLOW:
		return "GL_STACK_UNDERFLOW: This function would cause a stack underflow. The offending function is ignored, having no side effect other than to set the error flag.";
	case GL_OUT_OF_MEMORY:
		return "GL_OUT_OF_MEMORY: There is not enough memory left to execute the function. The state of OpenGL is undefined, except for the state of the error flags, after this error is recorded.";
	case GL_INVALID_FRAMEBUFFER_OPERATION:
		return "GL_INVALID_FRAMEBUFFER_OPERATION: The value of FRAMEBUFFER_STATUS is not FRAMEBUFFER_COMPLETE when any attempts to render to or read from the framebuffer are made.";
	default:
		return "Unknown GL-Error";
	}
	return 0; // unsicher, ist aber eh nur hier um den compiler zu beruhigen
}

void GLERRORimpl(const char *text, const char *filename, int linenr)
{
	GLenum e = glGetError();
	if (e != GL_NO_ERROR)
	{
		std::ostringstream location;
		location << text << '(' << filename << ':' << linenr << ')';

		const char *errortxt = OGLErrorToString(e);
		std::cout << "GL_ERROR in " << location.str() << ": " << errortxt;
	}
};

void GLAPIENTRY glDebugOutput(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar *message,
	const void *userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}


void enableDebug()
{
	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(glDebugOutput, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
}

////////////////////////////////////////////////////////////////////////////////
template<typename T, std::size_t N>
std::vector<T> toVector(T const (&a)[N])
{
	return std::vector<T>(std::begin(a), std::end(a));
}

////////////////////////////////////////////////////////////////////////////////
vk::UniqueShaderModule createShader(vk::Device dev, std::vector<std::uint32_t> const& spv)
{
	auto const shader_info{ vk::ShaderModuleCreateInfo{}
		.setCodeSize(spv.size() * sizeof(std::uint32_t))
		.setPCode(spv.data()) };
	return dev.createShaderModuleUnique(shader_info);
}

// chooses a memory type index based on memory requirements as well as preferred and required memory property flags

////////////////////////////////////////////////////////////////////////////////
uint32_t selectMemoryTypeIndex(
	vk::PhysicalDevice phys_dev,
	vk::MemoryRequirements mem_req,
	vk::MemoryPropertyFlags preferred,
	vk::MemoryPropertyFlags required)
{
	auto const mem_props{ phys_dev.getMemoryProperties() };
	for (unsigned i{ 0 }; i < VK_MAX_MEMORY_TYPES; ++i)
	{
		if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & preferred) == preferred)
			return i;
	}
	if (required != preferred)
	{
		for (unsigned i{ 0 }; i < VK_MAX_MEMORY_TYPES; ++i)
		{
			if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & required) == required)
				return i;
		}
	}

	throw std::runtime_error{ "required memory type not available" };
}

//////////////////////////////////////////////////////////////////////////////////

struct Image
{
	vk::UniqueImage image;
	vk::UniqueDeviceMemory memory;
	uint64_t memory_size;

	Image(vk::UniqueImage&& _image,vk::UniqueDeviceMemory&& _memory,uint64_t mem_size) :
		image{ std::move(_image) },
		memory(std::move(_memory)),
		memory_size(mem_size)
	{}

	Image()
	{}
};

struct Buffer
{
	vk::UniqueBuffer buffer;
	vk::UniqueDeviceMemory memory;

	Buffer(vk::UniqueBuffer&& _buffer, vk::UniqueDeviceMemory&& _memory) :
		buffer{ std::move(_buffer) },
		memory(std::move(_memory))
	{}

	Buffer()
	{}
};

class Scene
{

public:
	using Window = GLFWwindow;
	Scene();
	~Scene();
public:
	void initialize();
	void run();
	void shutdown();

private:
	void createWindowAndSurface();
	void initializeVKInstance();
	void selectQueueFamilyAndPhysicalDevice();
	void initializeDevice();
	
	///////////////////////////////////////////////////
	void createOffscreenImages();
	void createOffscreenImageViews();
	
	////////////////////////////////////////////////////////////////////
	// Renderer
	void createPass();

	void createFramebuffer(); // one frame buffer is enough for both render passes (they are compatible)
	void allocateCommandBuffer();

	// descriptors and pipeline layouts
	void createShaderInterface();

	void createPipeline();
	void createGeomAndMatrices(); // create few bounding boxes and setup vertex buffer with aabb ids
	void initSyncEntities();

	void buildCommandBuffer(uint32_t image_index);

private:
	// helper
	Image allocateImage(const vk::ImageCreateInfo& img_ci, vk::MemoryPropertyFlags required);
	Buffer allocateBuffer(const vk::BufferCreateInfo& b_ci, vk::MemoryPropertyFlags required);
	void copyBuffer(const Buffer& src,const Buffer& dst, vk::ArrayProxy<const vk::BufferCopy> copy_regions);

	void initOpenGLEntities();

private:
	Window* m_window;
	uint32_t m_width = 1920;
	uint32_t m_height = 1080;

	// Tweaker
	const vk::Format m_col_attachment_format = vk::Format::eR8G8B8A8Unorm;
	const vk::Format m_depth_image_format = vk::Format::eD32Sfloat;

	////////////////////////////////////////////////////////////////////////////////////////////////

	vk::UniqueInstance m_instance;
	vk::PhysicalDevice m_phys_dev;
	uint32_t m_gq_fam_idx = std::numeric_limits<uint32_t>::max(); // graphics queue fam index
	vk::UniqueDevice m_device;
	vk::Queue m_gr_queue;

	////////////////////////////////////////////////////
	// surface and swap chain
	Image m_color_buffer_image;
	Image m_depth_buffer_image;
	vk::UniqueImageView m_color_img_view;
	vk::UniqueImageView m_depth_image_view;

	/////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// Renderer 
	vk::UniqueRenderPass m_render_pass;
	std::vector<vk::UniqueFramebuffer> m_framebuffers;

	vk::UniqueCommandPool m_cmd_b_pool;

	// not really good way to do it but it is ok for bug reporting
	std::vector<vk::UniqueCommandBuffer> m_command_buffers;

	vk::UniqueShaderModule m_vert_shader;
	vk::UniqueShaderModule m_geom_shader;
	vk::UniqueShaderModule m_frag_shader;

	vk::UniqueDescriptorSetLayout m_ds_layout;
	vk::UniquePipelineLayout m_pipeline_layout;
	vk::UniqueDescriptorPool m_ds_pool;
	vk::UniqueDescriptorSet m_ds;

	vk::UniquePipeline m_pipeline;

	Buffer m_aabb_storage_buffer;
	Buffer m_aabb_node_id_buffer;

	std::vector<vk::UniqueFence> m_fences;
	vk::UniqueSemaphore m_gl_wait_semaphore;

	struct AABB
	{
		float center[4];
		float radius[4];
	};

	std::vector<AABB> m_bboxes;

	//GL PART
	GLuint m_fbo_id;
	GLuint m_col_texture_id = 0;
	GLuint m_dep_texture_id = 0;
	GLuint m_col_mem_id = 0;
	GLuint m_dep_mem_id = 0;
	GLuint m_sem_id = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Scene::Scene()
{

}

Scene::~Scene()
{
	if (m_window)
		glfwDestroyWindow(m_window);

	glfwTerminate();
}


void Scene::initialize()
{
	// Initializing window instance devices and swapchain

	try
	{
		createWindowAndSurface();

		auto context = wglGetCurrentContext();
		auto hdc = wglGetCurrentDC();

		wglMakeCurrent(hdc, 0); // need this with AMD Drivers (otherwise opengl context gets brocken)

		initializeVKInstance();
		selectQueueFamilyAndPhysicalDevice();
		initializeDevice();

		wglMakeCurrent(hdc, context); // reset the context ==> AMD workaround

		createOffscreenImages();
		createOffscreenImageViews();

		////////////////////////////////////////////////////////////

		// Initializing Renderer
		createPass();		
		createFramebuffer();
		allocateCommandBuffer();
		createGeomAndMatrices();
		createShaderInterface();
		createPipeline();
		initSyncEntities();

		//interop entities
		initOpenGLEntities();
	}
	catch (vk::SystemError& e)
	{
		throw std::runtime_error(e.what());
	}
}

void Scene::run()
{
	while (1)
	{
		glfwPollEvents();
		if (glfwWindowShouldClose(m_window))
			break;

		m_device->waitForFences(*m_fences[0], true, UINT64_MAX);
		m_device->resetFences(*m_fences[0]);

		buildCommandBuffer(0);

		const vk::PipelineStageFlags wait_mask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submit_info{};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &*m_command_buffers[0];
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &*m_gl_wait_semaphore;
		submit_info.waitSemaphoreCount = {};
		submit_info.pWaitDstStageMask = {};
		submit_info.pWaitSemaphores = {};


		m_gr_queue.submit(submit_info, *m_fences[0]);

		GLenum layout = GL_LAYOUT_TRANSFER_SRC_EXT;
		glWaitSemaphoreEXT(m_sem_id, 0, nullptr, 1, &m_col_texture_id, &layout);

		glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo_id);
		glReadBuffer(GL_COLOR_ATTACHMENT0);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glDrawBuffer(GL_BACK_LEFT);
		glBlitFramebuffer(0, 0, m_width, m_height, 0, 0, m_width, m_height, GL_COLOR_BUFFER_BIT, GL_LINEAR);

		SwapBuffers(wglGetCurrentDC());
	}
}

void Scene::shutdown()
{

}

void glfwerror(int ec, const char* emsg)
{
	std::cout << "Error Code: " << ec << " Error Msg: " << emsg << std::endl;
}

void Scene::createWindowAndSurface()
{
	glfwInit();
	glfwSetErrorCallback(glfwerror);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

	m_window = glfwCreateWindow(m_width,m_height,"",nullptr,nullptr);
	
	glfwMakeContextCurrent(m_window);
	glewInit();
	enableDebug();

	if (m_window == nullptr)
	{
		throw std::runtime_error("Window Creation failed!");
		glfwTerminate();
	}
}

void Scene::initializeVKInstance()
{
	std::vector<const char*> extensions;
	std::vector < const char*> layers;

	if (!isInstanceExtensionAvailable(VK_KHR_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");

	if (!isInstanceExtensionAvailable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");

	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

	if (isInstanceLayerAvailable("VK_LAYER_LUNARG_standard_validation"))
		layers.push_back("VK_LAYER_LUNARG_standard_validation");

	vk::InstanceCreateInfo inst_ci{};

	inst_ci.enabledLayerCount = (uint32_t)layers.size();
	inst_ci.ppEnabledLayerNames = layers.data();
	inst_ci.enabledExtensionCount = (uint32_t)extensions.size();
	inst_ci.ppEnabledExtensionNames = extensions.data();

	vk::ApplicationInfo app_info{};
	app_info.applicationVersion = VK_MAKE_VERSION(1,0,0);
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "Test Engine";
	app_info.apiVersion = VK_MAKE_VERSION(1, 1, 0);

	inst_ci.pApplicationInfo = &app_info;

	m_instance = vk::createInstanceUnique(inst_ci);
}

void Scene::selectQueueFamilyAndPhysicalDevice()
{
	// simplified version

// CHANGE ID TO SELECT PHYSICAL DEVICE
	const uint32_t required_phys_idx = 0;
	const auto phys_devs = m_instance->enumeratePhysicalDevices();

	if (phys_devs.size() <= required_phys_idx)
	{
		throw std::runtime_error("Invalid Physical Device Index provided!");
	}

	m_phys_dev = phys_devs[required_phys_idx];


	// I need only one graphics queue to show the bug ==> simply find first graphics queue family index with one queue

	const auto queue_fam_props = m_phys_dev.getQueueFamilyProperties();

	for (std::size_t i = 0;queue_fam_props.size();++i)
	{
		const auto prop = queue_fam_props[i];

		if (prop.queueFlags & vk::QueueFlagBits::eGraphics && prop.queueCount > 0)
		{
			m_gq_fam_idx = (uint32_t)i;
			break;
		}
	}

	if (m_gq_fam_idx == std::numeric_limits<uint32_t>::max())
	{
		// something very terrible is happened (device has no graphics queue)
		throw std::runtime_error("Can not find graphics family index!");
	}
}

void Scene::initializeDevice()
{
	vk::DeviceCreateInfo dev_ci{};

	vk::PhysicalDeviceFeatures dev_features{};
	dev_features.geometryShader                 = true;
	dev_features.robustBufferAccess             = true;
	dev_features.fragmentStoresAndAtomics       = true;
	dev_features.vertexPipelineStoresAndAtomics = true;

	vk::DeviceQueueCreateInfo dev_q_ci{};

	float queue_prio = 1.0f;

	dev_q_ci.queueCount = 1;
	dev_q_ci.pQueuePriorities = &queue_prio;
	dev_q_ci.queueFamilyIndex = m_gq_fam_idx;

	dev_ci.pEnabledFeatures = &dev_features;
	dev_ci.queueCreateInfoCount = 1;
	dev_ci.pQueueCreateInfos = &dev_q_ci;
	
	std::vector<const char*> extensions;

	extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
	extensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);

	dev_ci.enabledExtensionCount = (uint32_t)extensions.size();
	dev_ci.ppEnabledExtensionNames = extensions.data();

	m_device = m_phys_dev.createDeviceUnique(dev_ci);
	m_gr_queue = m_device->getQueue(m_gq_fam_idx, 0);
}

void Scene::createOffscreenImages()
{
	{
		vk::ImageCreateInfo im_ci{};
		im_ci.arrayLayers = 1;
		im_ci.extent = { m_width,m_height,1 };
		im_ci.format = m_col_attachment_format;
		im_ci.imageType = vk::ImageType::e2D;
		im_ci.initialLayout = vk::ImageLayout::eUndefined;
		im_ci.mipLevels = 1;
		im_ci.samples = vk::SampleCountFlagBits::e1;
		im_ci.tiling = vk::ImageTiling::eOptimal;
		im_ci.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;

		m_color_buffer_image = allocateImage(im_ci, vk::MemoryPropertyFlagBits::eDeviceLocal);
	}

	{
		// now create DepthBuffer Image

		vk::ImageCreateInfo im_ci{};
		im_ci.arrayLayers = 1;
		im_ci.extent = { m_width,m_height,1 };
		im_ci.format = m_depth_image_format;
		im_ci.imageType = vk::ImageType::e2D;
		im_ci.initialLayout = vk::ImageLayout::eUndefined;
		im_ci.mipLevels = 1;
		im_ci.samples = vk::SampleCountFlagBits::e1;
		im_ci.tiling = vk::ImageTiling::eOptimal;
		im_ci.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

		m_depth_buffer_image = allocateImage(im_ci, vk::MemoryPropertyFlagBits::eDeviceLocal);
	}
}

Image Scene::allocateImage(const vk::ImageCreateInfo& img_ci, vk::MemoryPropertyFlags required)
{

	// ALLCOATES EXPORTABLE MEMORY
	vk::UniqueImage img = m_device->createImageUnique(img_ci);
	const auto mem_req = m_device->getImageMemoryRequirements(*img);

	uint32_t index = selectMemoryTypeIndex(m_phys_dev, mem_req, required, required);

	vk::ExportMemoryAllocateInfo export_memory_alloc_info{ vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 };

	vk::MemoryAllocateInfo ma_i{};
	ma_i.pNext = &export_memory_alloc_info;
	ma_i.allocationSize = mem_req.size;
	ma_i.memoryTypeIndex = index;

	vk::UniqueDeviceMemory dev_mem = m_device->allocateMemoryUnique(ma_i);

	// Bind image to memory;
	m_device->bindImageMemory(*img, *dev_mem, 0);

	return{ std::move(img),std::move(dev_mem),mem_req.size};

}

Buffer Scene::allocateBuffer(const vk::BufferCreateInfo& b_ci, vk::MemoryPropertyFlags required)
{
	vk::UniqueBuffer buff = m_device->createBufferUnique(b_ci);
	const auto mem_req = m_device->getBufferMemoryRequirements(*buff);

	uint32_t index = selectMemoryTypeIndex(m_phys_dev, mem_req, required, required);

	vk::MemoryAllocateInfo ma_i{};
	ma_i.allocationSize = mem_req.size;
	ma_i.memoryTypeIndex = index;

	vk::UniqueDeviceMemory dev_mem = m_device->allocateMemoryUnique(ma_i);

	m_device->bindBufferMemory(*buff, *dev_mem, 0);

	return{std::move(buff),std::move(dev_mem)};
}

void Scene::copyBuffer(const Buffer& src, const Buffer& dst, vk::ArrayProxy<const vk::BufferCopy> copy_regions)
{
	vk::CommandPoolCreateInfo cmd_pool_ci{};
	cmd_pool_ci.queueFamilyIndex = m_gq_fam_idx;
	cmd_pool_ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

	auto cmd_pool = m_device->createCommandPoolUnique(cmd_pool_ci);

	vk::CommandBufferAllocateInfo cmd_b_ai{};
	cmd_b_ai.commandBufferCount = 1;
	cmd_b_ai.commandPool = *m_cmd_b_pool;
	cmd_b_ai.level = vk::CommandBufferLevel::ePrimary;

	auto cmd_buffer = m_device->allocateCommandBuffersUnique(cmd_b_ai);

	cmd_buffer[0]->begin(vk::CommandBufferBeginInfo());
	cmd_buffer[0]->copyBuffer(*src.buffer, *dst.buffer, copy_regions);
	cmd_buffer[0]->end();

	vk::SubmitInfo submit_info{};
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &*cmd_buffer[0];

	m_gr_queue.submit(submit_info, vk::Fence{});
	m_gr_queue.waitIdle();
}

void Scene::createOffscreenImageViews()
{
	vk::ImageSubresourceRange img_sb_range{};
	img_sb_range.aspectMask = vk::ImageAspectFlagBits::eColor;
	img_sb_range.levelCount = 1;
	img_sb_range.layerCount = 1;

	vk::ImageViewCreateInfo col_att_imgv_ci{};
	col_att_imgv_ci.subresourceRange = img_sb_range;
	col_att_imgv_ci.format = m_col_attachment_format;
	col_att_imgv_ci.viewType = vk::ImageViewType::e2D;

	col_att_imgv_ci.image = *m_color_buffer_image.image;
	m_color_img_view= m_device->createImageViewUnique(col_att_imgv_ci);
	

	// depth image view
	img_sb_range.aspectMask = vk::ImageAspectFlagBits::eDepth;
	col_att_imgv_ci.subresourceRange = img_sb_range;
	col_att_imgv_ci.format = m_depth_image_format;
	col_att_imgv_ci.image = *m_depth_buffer_image.image;

	m_depth_image_view = m_device->createImageViewUnique(col_att_imgv_ci);

}

void Scene::createPass()
{
	vk::AttachmentDescription color_att_desc{};
	color_att_desc.format      = m_col_attachment_format;
	color_att_desc.samples     = vk::SampleCountFlagBits::e1;
	color_att_desc.loadOp      = vk::AttachmentLoadOp::eClear;
	color_att_desc.storeOp     = vk::AttachmentStoreOp::eStore;
	color_att_desc.finalLayout = vk::ImageLayout::eTransferSrcOptimal;

	vk::AttachmentDescription depth_att_desc{};
	depth_att_desc.format       = m_depth_image_format;
	depth_att_desc.samples      = vk::SampleCountFlagBits::e1;
	depth_att_desc.loadOp       = vk::AttachmentLoadOp::eClear;
	depth_att_desc.storeOp      = vk::AttachmentStoreOp::eDontCare;
	depth_att_desc.finalLayout  = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	const std::array<vk::AttachmentDescription, 2> att_descs 
	{
		color_att_desc,
		depth_att_desc
	};

	vk::AttachmentReference color_att_ref{};
	color_att_ref.attachment = 0;
	color_att_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentReference depth_att_ref{};

	depth_att_ref.attachment = 1;
	depth_att_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::SubpassDescription subpass_desc{};
	subpass_desc.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass_desc.colorAttachmentCount = 1;
	subpass_desc.pColorAttachments = &color_att_ref;
	subpass_desc.pDepthStencilAttachment = &depth_att_ref;

	vk::RenderPassCreateInfo rp_ci{};

	rp_ci.attachmentCount = (uint32_t)att_descs.size();
	rp_ci.pAttachments = att_descs.data();
	rp_ci.subpassCount = 1;
	rp_ci.pSubpasses = &subpass_desc;

	m_render_pass = m_device->createRenderPassUnique(rp_ci);	
}


void Scene::createFramebuffer()
{
	std::array<vk::ImageView,2> attachments;
	attachments[0] = nullptr;
	attachments[1] = *m_depth_image_view;

	vk::FramebufferCreateInfo fb_ci{};
	fb_ci.renderPass = *m_render_pass;
	fb_ci.attachmentCount = (uint32_t)attachments.size();
	fb_ci.pAttachments = attachments.data();
	fb_ci.width = m_width;
	fb_ci.height = m_height;
	fb_ci.layers = 1;

	for (uint32_t i = 0;i < 1;++i)
	{
		attachments[0] = *m_color_img_view;
		m_framebuffers.push_back(m_device->createFramebufferUnique(fb_ci));
	}
}

void Scene::allocateCommandBuffer()
{
	vk::CommandPoolCreateInfo cmd_pool_ci{};
	cmd_pool_ci.queueFamilyIndex = m_gq_fam_idx;
	cmd_pool_ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

	m_cmd_b_pool = m_device->createCommandPoolUnique(cmd_pool_ci);

	vk::CommandBufferAllocateInfo cmd_b_ai{};
	cmd_b_ai.commandBufferCount = 3;
	cmd_b_ai.commandPool = *m_cmd_b_pool;
	cmd_b_ai.level = vk::CommandBufferLevel::ePrimary;
	
	m_command_buffers = m_device->allocateCommandBuffersUnique(cmd_b_ai);
}


void Scene::createShaderInterface()
{
	vk::DescriptorSetLayoutBinding ds_binding{};
	ds_binding.binding = 0;
	ds_binding.descriptorCount = 1;
	ds_binding.descriptorType = vk::DescriptorType::eStorageBuffer;
	ds_binding.stageFlags = vk::ShaderStageFlagBits::eAllGraphics;

	vk::DescriptorSetLayoutCreateInfo dsl_ci{};
	dsl_ci.bindingCount = 1;
	dsl_ci.pBindings = &ds_binding;
	
	m_ds_layout = m_device->createDescriptorSetLayoutUnique(dsl_ci);

	vk::PipelineLayoutCreateInfo pl_ci{};
	pl_ci.setLayoutCount = 1;
	pl_ci.pSetLayouts = &*m_ds_layout;

	std::vector<vk::PushConstantRange> ranges
	{
		{vk::ShaderStageFlagBits::eAllGraphics,0,sizeof(float) * 16 + sizeof(float) * 4 + sizeof(uint32_t) }, // proj mat + cam position + num_values
	};

	pl_ci.pushConstantRangeCount = (uint32_t)ranges.size();
	pl_ci.pPushConstantRanges = ranges.data();

	m_pipeline_layout = m_device->createPipelineLayoutUnique(pl_ci);

	vk::DescriptorPoolSize ds_p_size{};
	ds_p_size.type = vk::DescriptorType::eStorageBuffer;
	ds_p_size.descriptorCount = 1;

	vk::DescriptorPoolCreateInfo ds_ci{};
	ds_ci.maxSets = 1;
	ds_ci.poolSizeCount = 1;
	ds_ci.pPoolSizes = &ds_p_size;
	ds_ci.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;

	m_ds_pool = m_device->createDescriptorPoolUnique(ds_ci);

	vk::DescriptorSetAllocateInfo ds_alloc_info{};
	ds_alloc_info.descriptorPool = *m_ds_pool;
	ds_alloc_info.descriptorSetCount = 1;
	ds_alloc_info.pSetLayouts = &*m_ds_layout;
	
	auto result = m_device->allocateDescriptorSetsUnique(ds_alloc_info);
	m_ds = std::move(result[0]);

	vk::DescriptorBufferInfo bi{};
	bi.buffer = *m_aabb_storage_buffer.buffer;
	bi.offset = 0;
	bi.range = VK_WHOLE_SIZE;
	
	vk::WriteDescriptorSet write_set{};
	write_set.descriptorCount = 1;
	write_set.descriptorType = vk::DescriptorType::eStorageBuffer;
	write_set.dstBinding = 0;
	write_set.pBufferInfo = &bi;
	write_set.dstSet = *m_ds;
	m_device->updateDescriptorSets(write_set, {});
}

void Scene::createPipeline()
{
	//////////////////////////////////////////////////////////////////////////
	// Vertex Format

	vk::VertexInputAttributeDescription vt_att_desc{};
	vt_att_desc.format = vk::Format::eR32Uint;

	vk::VertexInputBindingDescription vt_bind_desc{};
	vt_bind_desc.inputRate = vk::VertexInputRate::eVertex;
	vt_bind_desc.stride = sizeof(uint32_t);

	vk::PipelineVertexInputStateCreateInfo vt_inp_ci{};
	vt_inp_ci.pVertexAttributeDescriptions = &vt_att_desc;
	vt_inp_ci.pVertexBindingDescriptions = &vt_bind_desc;
	vt_inp_ci.vertexAttributeDescriptionCount = 1;
	vt_inp_ci.vertexBindingDescriptionCount = 1;

	vk::PipelineColorBlendAttachmentState cbas_ci{};
	cbas_ci.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB;

	vk::PipelineColorBlendStateCreateInfo cbs_ci{};
	cbs_ci.attachmentCount = 1;
	cbs_ci.pAttachments = &cbas_ci;

	vk::PipelineDepthStencilStateCreateInfo dss_ci{};
	dss_ci.depthWriteEnable = VK_FALSE;
	dss_ci.depthTestEnable = VK_FALSE;

	vk::PipelineInputAssemblyStateCreateInfo as_ci{};
	as_ci.topology = vk::PrimitiveTopology::ePointList;

	vk::PipelineMultisampleStateCreateInfo mss_ci{};
	mss_ci.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineRasterizationStateCreateInfo rss_ci{};
	rss_ci.cullMode = vk::CullModeFlagBits::eNone;
	rss_ci.polygonMode = vk::PolygonMode::eFill;
	rss_ci.lineWidth = 1.0f;

	m_vert_shader = createShader(*m_device, toVector(::bbox_vert));
	m_geom_shader = createShader(*m_device, toVector(::bbox_geom));
	m_frag_shader = createShader(*m_device, toVector(::bbox_frag));

	std::vector <vk::PipelineShaderStageCreateInfo> sh_stages;
	
	vk::PipelineShaderStageCreateInfo ss_ci{};
	ss_ci.pName = "main";

	ss_ci.module = *m_vert_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eVertex;

	sh_stages.push_back(ss_ci);

	ss_ci.module = *m_geom_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eGeometry;

	sh_stages.push_back(ss_ci);

	ss_ci.module = *m_frag_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eFragment;

	sh_stages.push_back(ss_ci);

	vk::PipelineViewportStateCreateInfo vps_ci{};

	vk::Viewport viewport{};
	viewport.width = (float) m_width;
	viewport.height = (float) m_height;
	viewport.minDepth = 0.0f;
	viewport.maxDepth = 1.0f;

	vps_ci.viewportCount = 1;
	vps_ci.pViewports = &viewport;

	vk::Rect2D scissor{};
	scissor.extent.width = m_width;
	scissor.extent.height = m_height;

	vps_ci.scissorCount = 1;
	vps_ci.pScissors = &scissor;

	///////////////////////////////////////////////////////////////////////////
	vk::GraphicsPipelineCreateInfo gp_ci{};

	gp_ci.pVertexInputState = &vt_inp_ci;
	gp_ci.layout = *m_pipeline_layout;
	gp_ci.pColorBlendState = &cbs_ci;
	gp_ci.pDepthStencilState = &dss_ci;
	gp_ci.pInputAssemblyState = &as_ci;
	gp_ci.pMultisampleState = &mss_ci;
	gp_ci.pRasterizationState = &rss_ci;
	gp_ci.stageCount = (uint32_t)sh_stages.size();
	gp_ci.pStages = sh_stages.data();
	gp_ci.renderPass = *m_render_pass;
	gp_ci.subpass = 0;
	gp_ci.pViewportState = &vps_ci;

	m_pipeline = m_device->createGraphicsPipelineUnique({}, gp_ci);
}

void Scene::createGeomAndMatrices()
{
	float rad_size = 0.2f;
	glm::vec4 center(0, 0, 0, 1);
	glm::vec4 radius(rad_size, rad_size, rad_size,0);

	glm::mat4x4 trf(1.0f);

	trf = glm::translate(glm::mat4(1.0f), glm::vec3(-(radius.x + rad_size) * 5.0f,0, 0));

	center = trf * center;

	std::vector<uint32_t> aabb_ids;

	// Create Few Bounding boxes to be used as storage buffer in vertex shader
	// as input to geometry shader
	// If no error visible increase the value of num_iterations variable (maybe to 1000)
	const uint32_t num_iterations = 1000;

	for (uint32_t i = 0;i < num_iterations;++i)
	{
		for (uint32_t j = 0;j < num_iterations;++j)
		{
			trf = glm::translate(glm::mat4(1.0f), glm::vec3((radius.x + rad_size * 2.0f) * j, (radius.x + rad_size* 2.0f) * i, -18.0));
			glm::vec4 result = trf * center;
			AABB aabb;

			aabb.center[0] = result.x;
			aabb.center[1] = result.y;
			aabb.center[2] = result.z;
			aabb.radius[0] = radius.x;
			aabb.radius[1] = radius.y;
			aabb.radius[2] = radius.z;

			m_bboxes.push_back(aabb);

			// create id vector from zero to num boxes (here 11 * 11)
			aabb_ids.push_back((uint32_t)m_bboxes.size() - 1);
		}
	}

	// now create storage buffer and copy aabbs to it

	vk::BufferCreateInfo b_ci{};
	
	b_ci.size = sizeof(AABB) * m_bboxes.size();
	b_ci.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;

	m_aabb_storage_buffer = allocateBuffer(b_ci, vk::MemoryPropertyFlagBits::eDeviceLocal);


	// create staging buffer and copy result to the main storage buffer;

	b_ci.usage = vk::BufferUsageFlagBits::eTransferSrc;
	Buffer staging = allocateBuffer(b_ci, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	void* mapped_memory = m_device->mapMemory(*staging.memory, 0, VK_WHOLE_SIZE);

	std::memcpy(mapped_memory, m_bboxes.data(), sizeof(AABB) * m_bboxes.size());
	m_device->unmapMemory(*staging.memory);

	vk::BufferCopy buffer_copy{};
	buffer_copy.dstOffset = 0;
	buffer_copy.srcOffset = 0;
	buffer_copy.size = sizeof(AABB) * m_bboxes.size();

	copyBuffer(staging, m_aabb_storage_buffer, buffer_copy);

	b_ci.size = sizeof(uint32_t) * aabb_ids.size();
	b_ci.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc;

	m_aabb_node_id_buffer = allocateBuffer(b_ci, vk::MemoryPropertyFlagBits::eDeviceLocal);
	b_ci.usage = vk::BufferUsageFlagBits::eTransferSrc;
	staging = allocateBuffer(b_ci, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

	mapped_memory = m_device->mapMemory(*staging.memory, 0, VK_WHOLE_SIZE);

	std::memcpy(mapped_memory, aabb_ids.data(), sizeof(uint32_t) * aabb_ids.size());
	m_device->unmapMemory(*staging.memory);

	buffer_copy.size = sizeof(uint32_t) * aabb_ids.size();
	copyBuffer(staging, m_aabb_node_id_buffer, buffer_copy);

}

void Scene::initSyncEntities()
{
	vk::FenceCreateInfo f_ci{};
	f_ci.flags = vk::FenceCreateFlagBits::eSignaled;

	for (uint32_t i = 0;i < 1; ++i)
	{
		m_fences.push_back(m_device->createFenceUnique(f_ci));
	}

	vk::SemaphoreCreateInfo s_ci{};
	vk::ExportSemaphoreCreateInfo es_ci{};
	s_ci.pNext = &es_ci;
	es_ci.handleTypes = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;;
	m_gl_wait_semaphore = m_device->createSemaphoreUnique(s_ci);
}

void Scene::buildCommandBuffer(uint32_t image_index)
{
	vk::CommandBufferBeginInfo cmd_begin_info{};
	auto& cmd = *m_command_buffers[image_index];

	cmd.begin(cmd_begin_info);
	const std::array<float,4> clear_color{0,0,1,1};

	const std::array<vk::ClearValue, 2> clear_values
	{
		vk::ClearColorValue(clear_color),
		vk::ClearDepthStencilValue(1,0)
	};

	vk::RenderPassBeginInfo rp_begin_info{};
	rp_begin_info.framebuffer = *m_framebuffers[image_index];
	rp_begin_info.renderArea = vk::Rect2D({ 0,0 }, { m_width,m_height });
	rp_begin_info.renderPass = *m_render_pass;
	rp_begin_info.clearValueCount = (uint32_t)clear_values.size();
	rp_begin_info.pClearValues = clear_values.data();

	cmd.beginRenderPass(rp_begin_info, vk::SubpassContents::eInline);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipeline_layout, 0, *m_ds, {});

	uint32_t size_of_pc = sizeof(float) * 16 + sizeof(float) * 4 + sizeof(uint32_t);
	std::vector<char> data(size_of_pc);

	glm::mat4 proj = glm::perspective(50.0f, (float)m_width / (float)m_height, 0.1f, 100000.0f);
	glm::vec4 cam_pos(0, 0, -18, 1);
	uint32_t num_nodes = (uint32_t)m_bboxes.size();

	uint32_t offset = 0;
	std::memcpy(&data[offset], glm::value_ptr(proj), sizeof(float) * 16);
	offset += sizeof(float) * 16;
	std::memcpy(&data[offset], glm::value_ptr(cam_pos), sizeof(float) * 4);
	offset += sizeof(float) * 4;
	std::memcpy(&data[offset], &num_nodes, sizeof(uint32_t));

	cmd.pushConstants(*m_pipeline_layout, vk::ShaderStageFlagBits::eAllGraphics, 0, uint32_t(data.size()),data.data());

	cmd.bindVertexBuffers(0, *m_aabb_node_id_buffer.buffer, { 0 });
	cmd.draw(uint32_t(m_bboxes.size()), 1, 0, 0);

	cmd.endRenderPass();
	cmd.end();
}

void Scene::initOpenGLEntities()
{
	//IMPORT MEMORY TO OPENGL
	vk::DispatchLoaderDynamic dyn_loader{ *m_instance, &vkGetInstanceProcAddr, *m_device, &vkGetDeviceProcAddr };
	glClipControl(GL_UPPER_LEFT, GL_ZERO_TO_ONE);
	glCreateTextures(GL_TEXTURE_2D, 1, &m_col_texture_id);
	glTextureParameteri(m_col_texture_id, GL_TEXTURE_TILING_EXT, GL_OPTIMAL_TILING_EXT);
	glCreateMemoryObjectsEXT(1, &m_col_mem_id);

	HANDLE  win32_handle = m_device->getMemoryWin32HandleKHR({ *m_color_buffer_image.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 }, dyn_loader);

	glImportMemoryWin32HandleEXT(m_col_mem_id, m_color_buffer_image.memory_size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32_handle);
	glTextureStorageMem2DEXT(m_col_texture_id, 1, GL_RGBA8, m_width, m_height, m_col_mem_id, 0);

	glCreateTextures(GL_TEXTURE_2D, 1, &m_dep_texture_id);
	glTextureParameteri(m_dep_texture_id, GL_TEXTURE_TILING_EXT, GL_OPTIMAL_TILING_EXT);
	glCreateMemoryObjectsEXT(1, &m_dep_mem_id);
	
	win32_handle = m_device->getMemoryWin32HandleKHR({ *m_depth_buffer_image.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32 }, dyn_loader);
	glImportMemoryWin32HandleEXT(m_dep_mem_id, m_depth_buffer_image.memory_size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32_handle);
	glTextureStorageMem2DEXT(m_dep_texture_id, 1, GL_DEPTH_COMPONENT32F, m_width, m_height, m_dep_mem_id, 0);

	glGenSemaphoresEXT(1, &m_sem_id);
	
	win32_handle = m_device->getSemaphoreWin32HandleKHR({ *m_gl_wait_semaphore, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32 }, dyn_loader);
	glImportSemaphoreWin32HandleEXT(m_sem_id, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, win32_handle);

	glCreateFramebuffers(1, &m_fbo_id);
	glNamedFramebufferTexture(m_fbo_id, GL_COLOR_ATTACHMENT0, m_col_texture_id, 0);
}

int main()
{
	SetProcessDPIAware();
	
	Scene scene;
	try
	{
		scene.initialize();
		scene.run();

	}
	catch (std::exception& e)
	{
		scene.shutdown();
		std::cout << "Error Occurred: " << e.what() << std::endl;
	}

	return 0;
}
