#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>
#include <zmq.h>
// #include <json-c/json.h>
#include <jansson.h>
#include "utils.h"

typedef struct
{
    float x1;
    float y1;
    float x2;
    float y2;
    float prob;
    int class_id;
    const char *class_name;
    int image_width;
    int image_height;
} BBox;

typedef struct
{
    BBox bboxes[1000];
    size_t num_boxes;
    // unsigned long frame_index;
} BBoxes;

json_t *load_json(const char *text)
{
    json_t *root;
    json_error_t error;

    root = json_loads(text, 0, &error);

    if (root)
    {
        return root;
    }
    else
    {
        fprintf(stderr, "json error on line %d: %s\n", error.line, error.text);
        return (json_t *)0;
    }
}

BBoxes parse_bboxes(json_t *element)
{
    json_t *bboxes_json = json_object_get(element, "bboxes");

    // Check to make sure that the bboxes is the right type
    assert(json_typeof(bboxes_json) == JSON_ARRAY);

    size_t i;
    size_t size = json_array_size(bboxes_json);
    BBoxes result;
    result.num_boxes = size;
    for (i = 0; i < size; i++)
    {
        json_t *bbox_node = json_array_get(bboxes_json, i);

        // Check to make sure that the node is a Dict/Map
        assert(json_typeof(bbox_node) == JSON_OBJECT);

        // Unpack the data from the json data structure
        float x1 = (float)json_real_value(json_object_get(bbox_node, "x1"));
        float y1 = (float)json_real_value(json_object_get(bbox_node, "y1"));
        float x2 = (float)json_real_value(json_object_get(bbox_node, "x2"));
        float y2 = (float)json_real_value(json_object_get(bbox_node, "y2"));
        float prob = (float)json_real_value(json_object_get(bbox_node, "prob"));

        const char *class_name = json_string_value(json_object_get(bbox_node, "object_class_name"));

        int class_id = json_integer_value(json_object_get(bbox_node, "object_class_id"));
        int image_width = json_integer_value(json_object_get(bbox_node, "img_width"));
        int image_height = json_integer_value(json_object_get(bbox_node, "img_height"));

        // Put it into the BBox struct
        BBox bbox = {
            x1, y1, x2, y2, prob, class_id, class_name, image_width, image_height};
        result.bboxes[i] = bbox;
    }

    return result;
}

void parse_classnames(json_t *element)
{
    json_t *classnames = json_object_get(element, "class_names");
    printf("Class Names:");
    print_json(classnames);
}

//  Modifed from: https://github.com/booksbyus/zguide/blob/master/examples/C/zhelpers.h
//  Receive 0MQ string from socket and convert into C string
//  Caller must free returned string. Returns NULL if the context
//  is being terminated.
static char *
json_recv(void *socket)
{
    enum
    {
        cap = 4096
    };
    char buffer[cap];
    int size = zmq_recv(socket, buffer, cap - 1, 0);
    if (size == -1)
        return NULL;
    buffer[size < cap ? size : cap - 1] = '\0';

    return strndup(buffer, sizeof(buffer) - 1);

    // remember that the strdup family of functions use malloc/alloc for space for the new string.  It must be manually
    // freed when you are done with it.  Failure to do so will allow a heap attack.
}

int main()
{
    void *context = zmq_ctx_new();
    void *subscriber = zmq_socket(context, ZMQ_SUB);
    int rc = zmq_connect(subscriber, "tcp://127.0.0.1:5001");
    assert(rc == 0);
    rc = zmq_setsockopt(subscriber, ZMQ_SUBSCRIBE, "", 0);
    assert(rc == 0);

    while (1)
    {
        char *topic = json_recv(subscriber);
        char *msg = json_recv(subscriber);

        if (strcmp(topic, "bboxes") == 0)
        {
            json_t *json_blob = load_json(msg);
            if (json_blob)
            {

                BBoxes results = parse_bboxes(json_blob);
                int n_boxes = (int)results.num_boxes;

                for (size_t i = 0; i < results.num_boxes; i++)
                {
                    // NOTE: THIS IS Where you can store the boxes and do stuff with them.
                    printf("X1: %f  Y1: %f X2: %f Y2: %f Prob: %f Class: %s\r\n", results.bboxes[i].x1, results.bboxes[i].y1, results.bboxes[i].x2, results.bboxes[i].y2, results.bboxes[i].prob, results.bboxes[i].class_name);
                }
                printf("\r\n\r\n\r\n");
                json_decref(json_blob);
            }
        }
        free(topic);
        free(msg);
    }

    zmq_close(subscriber);
    zmq_ctx_destroy(context);

    return 0;
}