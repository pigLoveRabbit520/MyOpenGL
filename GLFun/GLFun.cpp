#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "tiny_gltf.h"
#include <iostream>
#include <cmath>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

#define M_PI   3.14159265358979323846
#define BUFFER_OFFSET(i) ((char *)NULL + (i))
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main()\n"
"{\n"
"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
"}\0";
const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
"   FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);\n"
"}\n\0";

typedef struct {
    GLuint vb;
} GLBufferState;

typedef struct {
    std::vector<GLuint> diffuseTex;  // for each primitive in mesh
} GLMeshState;

typedef struct {
    std::map<std::string, GLint> attribs;
    std::map<std::string, GLint> uniforms;
} GLProgramState;

typedef struct {
    GLuint vb;     // vertex buffer
    size_t count;  // byte count
} GLCurvesState;

std::map<int, GLBufferState> gBufferState;
std::map<std::string, GLMeshState> gMeshState;
std::map<int, GLCurvesState> gCurvesMesh;
GLProgramState gGLProgramState;

void CheckErrors(std::string desc) {
    GLenum e = glGetError();
    if (e != GL_NO_ERROR) {
        fprintf(stderr, "OpenGL error in \"%s\": %d (%d)\n", desc.c_str(), e, e);
        exit(20);
    }
}

static size_t ComponentTypeByteSize(int type) {
    switch (type) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
    case TINYGLTF_COMPONENT_TYPE_BYTE:
        return sizeof(char);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
    case TINYGLTF_COMPONENT_TYPE_SHORT:
        return sizeof(short);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
    case TINYGLTF_COMPONENT_TYPE_INT:
        return sizeof(int);
    case TINYGLTF_COMPONENT_TYPE_FLOAT:
        return sizeof(float);
    case TINYGLTF_COMPONENT_TYPE_DOUBLE:
        return sizeof(double);
    default:
        return 0;
    }
}


static void SetupMeshState(tinygltf::Model& model, GLuint progId) {
    // Buffer
    {
        for (size_t i = 0; i < model.bufferViews.size(); i++) {
            const tinygltf::BufferView& bufferView = model.bufferViews[i];
            if (bufferView.target == 0) {
                std::cout << "WARN: bufferView.target is zero" << std::endl;
                continue;  // Unsupported bufferView.
            }

            int sparse_accessor = -1;
            for (size_t a_i = 0; a_i < model.accessors.size(); ++a_i) {
                const auto& accessor = model.accessors[a_i];
                if (accessor.bufferView == i) {
                    std::cout << i << " is used by accessor " << a_i << std::endl;
                    if (accessor.sparse.isSparse) {
                        std::cout
                            << "WARN: this bufferView has at least one sparse accessor to "
                            "it. We are going to load the data as patched by this "
                            "sparse accessor, not the original data"
                            << std::endl;
                        sparse_accessor = a_i;
                        break;
                    }
                }
            }

            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
            GLBufferState state;
            glGenBuffers(1, &state.vb);
            glBindBuffer(bufferView.target, state.vb);
            std::cout << "buffer.size= " << buffer.data.size()
                << ", byteOffset = " << bufferView.byteOffset << std::endl;

            if (sparse_accessor < 0)
                glBufferData(bufferView.target, bufferView.byteLength,
                    &buffer.data.at(0) + bufferView.byteOffset,
                    GL_STATIC_DRAW);
            else {
                const auto accessor = model.accessors[sparse_accessor];
                // copy the buffer to a temporary one for sparse patching
                unsigned char* tmp_buffer = new unsigned char[bufferView.byteLength];
                memcpy(tmp_buffer, buffer.data.data() + bufferView.byteOffset,
                    bufferView.byteLength);

                const size_t size_of_object_in_buffer =
                    ComponentTypeByteSize(accessor.componentType);
                const size_t size_of_sparse_indices =
                    ComponentTypeByteSize(accessor.sparse.indices.componentType);

                const auto& indices_buffer_view =
                    model.bufferViews[accessor.sparse.indices.bufferView];
                const auto& indices_buffer = model.buffers[indices_buffer_view.buffer];

                const auto& values_buffer_view =
                    model.bufferViews[accessor.sparse.values.bufferView];
                const auto& values_buffer = model.buffers[values_buffer_view.buffer];

                for (size_t sparse_index = 0; sparse_index < accessor.sparse.count;
                    ++sparse_index) {
                    int index = 0;
                    // std::cout << "accessor.sparse.indices.componentType = " <<
                    // accessor.sparse.indices.componentType << std::endl;
                    switch (accessor.sparse.indices.componentType) {
                    case TINYGLTF_COMPONENT_TYPE_BYTE:
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        index = (int)*(
                            unsigned char*)(indices_buffer.data.data() +
                                indices_buffer_view.byteOffset +
                                accessor.sparse.indices.byteOffset +
                                (sparse_index * size_of_sparse_indices));
                        break;
                    case TINYGLTF_COMPONENT_TYPE_SHORT:
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        index = (int)*(
                            unsigned short*)(indices_buffer.data.data() +
                                indices_buffer_view.byteOffset +
                                accessor.sparse.indices.byteOffset +
                                (sparse_index * size_of_sparse_indices));
                        break;
                    case TINYGLTF_COMPONENT_TYPE_INT:
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                        index = (int)*(
                            unsigned int*)(indices_buffer.data.data() +
                                indices_buffer_view.byteOffset +
                                accessor.sparse.indices.byteOffset +
                                (sparse_index * size_of_sparse_indices));
                        break;
                    }
                    std::cout << "updating sparse data at index  : " << index
                        << std::endl;
                    // index is now the target of the sparse index to patch in
                    const unsigned char* read_from =
                        values_buffer.data.data() +
                        (values_buffer_view.byteOffset +
                            accessor.sparse.values.byteOffset) +
                        (sparse_index * (size_of_object_in_buffer * accessor.type));

                    /*
                    std::cout << ((float*)read_from)[0] << "\n";
                    std::cout << ((float*)read_from)[1] << "\n";
                    std::cout << ((float*)read_from)[2] << "\n";
                    */

                    unsigned char* write_to =
                        tmp_buffer + index * (size_of_object_in_buffer * accessor.type);

                    memcpy(write_to, read_from, size_of_object_in_buffer * accessor.type);
                }

                // debug:
                /*for(size_t p = 0; p < bufferView.byteLength/sizeof(float); p++)
                {
                  float* b = (float*)tmp_buffer;
                  std::cout << "modified_buffer [" << p << "] = " << b[p] << '\n';
                }*/

                glBufferData(bufferView.target, bufferView.byteLength, tmp_buffer,
                    GL_STATIC_DRAW);
                delete[] tmp_buffer;
            }
            glBindBuffer(bufferView.target, 0);

            gBufferState[i] = state;
        }
    }

#if 0  // TODO(syoyo): Implement
    // Texture
    {
        for (size_t i = 0; i < model.meshes.size(); i++) {
            const tinygltf::Mesh& mesh = model.meshes[i];

            gMeshState[mesh.name].diffuseTex.resize(mesh.primitives.size());
            for (size_t primId = 0; primId < mesh.primitives.size(); primId++) {
                const tinygltf::Primitive& primitive = mesh.primitives[primId];

                gMeshState[mesh.name].diffuseTex[primId] = 0;

                if (primitive.material < 0) {
                    continue;
                }
                tinygltf::Material& mat = model.materials[primitive.material];
                // printf("material.name = %s\n", mat.name.c_str());
                if (mat.values.find("diffuse") != mat.values.end()) {
                    std::string diffuseTexName = mat.values["diffuse"].string_value;
                    if (model.textures.find(diffuseTexName) != model.textures.end()) {
                        tinygltf::Texture& tex = model.textures[diffuseTexName];
                        if (scene.images.find(tex.source) != model.images.end()) {
                            tinygltf::Image& image = model.images[tex.source];
                            GLuint texId;
                            glGenTextures(1, &texId);
                            glBindTexture(tex.target, texId);
                            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                            glTexParameterf(tex.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                            glTexParameterf(tex.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

                            // Ignore Texture.fomat.
                            GLenum format = GL_RGBA;
                            if (image.component == 3) {
                                format = GL_RGB;
                            }
                            glTexImage2D(tex.target, 0, tex.internalFormat, image.width,
                                image.height, 0, format, tex.type,
                                &image.image.at(0));

                            CheckErrors("texImage2D");
                            glBindTexture(tex.target, 0);

                            printf("TexId = %d\n", texId);
                            gMeshState[mesh.name].diffuseTex[primId] = texId;
                        }
                    }
                }
            }
        }
    }
#endif

    glUseProgram(progId);
    GLint vtloc = glGetAttribLocation(progId, "in_vertex");
    GLint nrmloc = glGetAttribLocation(progId, "in_normal");
    GLint uvloc = glGetAttribLocation(progId, "in_texcoord");

    // GLint diffuseTexLoc = glGetUniformLocation(progId, "diffuseTex");
    GLint isCurvesLoc = glGetUniformLocation(progId, "uIsCurves");

    gGLProgramState.attribs["POSITION"] = vtloc;
    gGLProgramState.attribs["NORMAL"] = nrmloc;
    gGLProgramState.attribs["TEXCOORD_0"] = uvloc;
    // gGLProgramState.uniforms["diffuseTex"] = diffuseTexLoc;
    gGLProgramState.uniforms["isCurvesLoc"] = isCurvesLoc;
};

static void DrawMesh(tinygltf::Model& model, const tinygltf::Mesh& mesh) {
    //// Skip curves primitive.
    // if (gCurvesMesh.find(mesh.name) != gCurvesMesh.end()) {
    //  return;
    //}

    // if (gGLProgramState.uniforms["diffuseTex"] >= 0) {
    //  glUniform1i(gGLProgramState.uniforms["diffuseTex"], 0);  // TEXTURE0
    //}

    if (gGLProgramState.uniforms["isCurvesLoc"] >= 0) {
        glUniform1i(gGLProgramState.uniforms["isCurvesLoc"], 0);
    }

    for (size_t i = 0; i < mesh.primitives.size(); i++) {
        const tinygltf::Primitive& primitive = mesh.primitives[i];

        if (primitive.indices < 0) return;

        // Assume TEXTURE_2D target for the texture object.
        // glBindTexture(GL_TEXTURE_2D, gMeshState[mesh.name].diffuseTex[i]);

        std::map<std::string, int>::const_iterator it(primitive.attributes.begin());
        std::map<std::string, int>::const_iterator itEnd(
            primitive.attributes.end());

        for (; it != itEnd; it++) {
            assert(it->second >= 0);
            const tinygltf::Accessor& accessor = model.accessors[it->second];
            glBindBuffer(GL_ARRAY_BUFFER, gBufferState[accessor.bufferView].vb);
            CheckErrors("bind buffer");
            int size = 1;
            if (accessor.type == TINYGLTF_TYPE_SCALAR) {
                size = 1;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC2) {
                size = 2;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC3) {
                size = 3;
            }
            else if (accessor.type == TINYGLTF_TYPE_VEC4) {
                size = 4;
            }
            else {
                assert(0);
            }
            // it->first would be "POSITION", "NORMAL", "TEXCOORD_0", ...
            if ((it->first.compare("POSITION") == 0) ||
                (it->first.compare("NORMAL") == 0) ||
                (it->first.compare("TEXCOORD_0") == 0)) {
                if (gGLProgramState.attribs[it->first] >= 0) {
                    // Compute byteStride from Accessor + BufferView combination.
                    int byteStride =
                        accessor.ByteStride(model.bufferViews[accessor.bufferView]);
                    assert(byteStride != -1);
                    glVertexAttribPointer(gGLProgramState.attribs[it->first], size,
                        accessor.componentType,
                        accessor.normalized ? GL_TRUE : GL_FALSE,
                        byteStride, BUFFER_OFFSET(accessor.byteOffset));
                    CheckErrors("vertex attrib pointer");
                    glEnableVertexAttribArray(gGLProgramState.attribs[it->first]);
                    CheckErrors("enable vertex attrib array");
                }
            }
        }

        const tinygltf::Accessor& indexAccessor =
            model.accessors[primitive.indices];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,
            gBufferState[indexAccessor.bufferView].vb);
        CheckErrors("bind buffer");
        int mode = -1;
        if (primitive.mode == TINYGLTF_MODE_TRIANGLES) {
            mode = GL_TRIANGLES;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_STRIP) {
            mode = GL_TRIANGLE_STRIP;
        }
        else if (primitive.mode == TINYGLTF_MODE_TRIANGLE_FAN) {
            mode = GL_TRIANGLE_FAN;
        }
        else if (primitive.mode == TINYGLTF_MODE_POINTS) {
            mode = GL_POINTS;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE) {
            mode = GL_LINES;
        }
        else if (primitive.mode == TINYGLTF_MODE_LINE_LOOP) {
            mode = GL_LINE_LOOP;
        }
        else {
            assert(0);
        }
        glDrawElements(mode, indexAccessor.count, indexAccessor.componentType,
            BUFFER_OFFSET(indexAccessor.byteOffset));
        CheckErrors("draw elements");

        {
            std::map<std::string, int>::const_iterator it(
                primitive.attributes.begin());
            std::map<std::string, int>::const_iterator itEnd(
                primitive.attributes.end());

            for (; it != itEnd; it++) {
                if ((it->first.compare("POSITION") == 0) ||
                    (it->first.compare("NORMAL") == 0) ||
                    (it->first.compare("TEXCOORD_0") == 0)) {
                    if (gGLProgramState.attribs[it->first] >= 0) {
                        glDisableVertexAttribArray(gGLProgramState.attribs[it->first]);
                    }
                }
            }
        }
    }
}

#if 0  // TODO(syoyo): Implement
static void DrawCurves(tinygltf::Scene& scene, const tinygltf::Mesh& mesh) {
    (void)scene;

    if (gCurvesMesh.find(mesh.name) == gCurvesMesh.end()) {
        return;
    }

    if (gGLProgramState.uniforms["isCurvesLoc"] >= 0) {
        glUniform1i(gGLProgramState.uniforms["isCurvesLoc"], 1);
    }

    GLCurvesState& state = gCurvesMesh[mesh.name];

    if (gGLProgramState.attribs["POSITION"] >= 0) {
        glBindBuffer(GL_ARRAY_BUFFER, state.vb);
        glVertexAttribPointer(gGLProgramState.attribs["POSITION"], 3, GL_FLOAT,
            GL_FALSE, /* stride */ 0, BUFFER_OFFSET(0));
        CheckErrors("curve: vertex attrib pointer");
        glEnableVertexAttribArray(gGLProgramState.attribs["POSITION"]);
        CheckErrors("curve: enable vertex attrib array");
    }

    glDrawArrays(GL_LINES, 0, state.count);

    if (gGLProgramState.attribs["POSITION"] >= 0) {
        glDisableVertexAttribArray(gGLProgramState.attribs["POSITION"]);
    }
}
#endif


static void QuatToAngleAxis(const std::vector<double> quaternion,
    double& outAngleDegrees,
    double* axis) {
    double qx = quaternion[0];
    double qy = quaternion[1];
    double qz = quaternion[2];
    double qw = quaternion[3];

    double angleRadians = 2 * acos(qw);
    if (angleRadians == 0.0) {
        outAngleDegrees = 0.0;
        axis[0] = 0.0;
        axis[1] = 0.0;
        axis[2] = 1.0;
        return;
    }

    double denom = sqrt(1 - qw * qw);
    outAngleDegrees = angleRadians * 180.0 / M_PI;
    axis[0] = qx / denom;
    axis[1] = qy / denom;
    axis[2] = qz / denom;
}

// Hierarchically draw nodes
static void DrawNode(tinygltf::Model& model, const tinygltf::Node& node) {
    // Apply xform

    glPushMatrix();
    if (node.matrix.size() == 16) {
        // Use `matrix' attribute
        glMultMatrixd(node.matrix.data());
    }
    else {
        // Assume Trans x Rotate x Scale order
        if (node.translation.size() == 3) {
            glTranslated(node.translation[0], node.translation[1],
                node.translation[2]);
        }

        if (node.rotation.size() == 4) {
            double angleDegrees;
            double axis[3];

            QuatToAngleAxis(node.rotation, angleDegrees, axis);

            glRotated(angleDegrees, axis[0], axis[1], axis[2]);
        }

        if (node.scale.size() == 3) {
            glScaled(node.scale[0], node.scale[1], node.scale[2]);
        }
    }

    // std::cout << "node " << node.name << ", Meshes " << node.meshes.size() <<
    // std::endl;

    // std::cout << it->first << std::endl;
    // FIXME(syoyo): Refactor.
    // DrawCurves(scene, it->second);
    if (node.mesh > -1) {
        assert(node.mesh < model.meshes.size());
        DrawMesh(model, model.meshes[node.mesh]);
    }

    // Draw child nodes.
    for (size_t i = 0; i < node.children.size(); i++) {
        assert(node.children[i] < model.nodes.size());
        DrawNode(model, model.nodes[node.children[i]]);
    }

    glPopMatrix();
}

static void DrawModel(tinygltf::Model& model) {
#if 0
    std::map<std::string, tinygltf::Mesh>::const_iterator it(scene.meshes.begin());
    std::map<std::string, tinygltf::Mesh>::const_iterator itEnd(scene.meshes.end());

    for (; it != itEnd; it++) {
        DrawMesh(scene, it->second);
        DrawCurves(scene, it->second);
    }
#else
    // If the glTF asset has at least one scene, and doesn't define a default one
    // just show the first one we can find
    assert(model.scenes.size() > 0);
    int scene_to_display = model.defaultScene > -1 ? model.defaultScene : 0;
    const tinygltf::Scene& scene = model.scenes[scene_to_display];
    for (size_t i = 0; i < scene.nodes.size(); i++) {
        DrawNode(model, model.nodes[scene.nodes[i]]);
    }
#endif
}


int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    // build and compile our shader program
    // ------------------------------------
    // vertex shader
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    // check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // fragment shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    // check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }
    // link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    // check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
         0.5f,  0.5f, 0.0f,  // top right
         0.5f, -0.5f, 0.0f,  // bottom right
        -0.5f, -0.5f, 0.0f,  // bottom left
        -0.5f,  0.5f, 0.0f   // top left 
    };
    unsigned int indices[] = {  // note that we start from 0!
        0, 1, 3,  // first Triangle
        1, 2, 3   // second Triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
    glBindVertexArray(0);


    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // draw our first triangle
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        //glDrawArrays(GL_TRIANGLES, 0, 6);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        // glBindVertexArray(0); // no need to unbind it every time 

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}