!pip -q install lucid>=0.3.8 --upgrade
!npm install -g svelte-cli@2.2.0

import numpy as np
import tensorflow as tf

from lucid.modelzoo.vision_base import Model
from lucid.misc.io import show
import lucid.optvis.render as render
from lucid.misc.io import show, load
from lucid.misc.io.reading import read
from lucid.misc.io.showing import _image_url
from lucid.misc.gradient_override import gradient_override_map
import lucid.scratch.web.svelte as lucid_svelte

class NasNetMobile(Model):
  model_path = 'keras_to_tf_model.pb'
  image_shape = [None, 400, 400, 3]
  image_value_range = (0, 1)
  input_name = 'conv2d_1_input:0'
  
model = NasNetMobile()
model.load_graphdef()

for node in model.graph_def.node:
    print(node.name)

labels = ['breakfast', 'dinner', 'work', 'clean', 'cook', 'exercise', 'play toys', 'play games', 'recreation','no activity']
print(labels)

def raw_class_spatial_attr(img, layer, label, override=None):
  """How much did spatial positions at a given layer effect a output class?"""

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
    t_input = tf.placeholder_with_default(img, [None, None, None, 3])
    T = render.import_model(model, t_input, t_input)
    
    # Compute activations
    acts = T(layer).eval()
#     print(acts)
    
    # Added this
#     acts = acts*1000
    
    if label is None: return np.zeros(acts.shape[1:-1])
    
    # Compute gradient
    score = T("dense_3/Softmax")[0, labels.index(label)]
    t_grad = tf.gradients([score], [T(layer)])[0]   
    grad = t_grad.eval({T(layer) : acts})
    # Added this
    #print(grad)
    grad = grad*(10**9.00)
    
    # Linear approximation of effect of spatial position
    return np.sum(acts * grad, -1)[0]


def raw_spatial_spatial_attr(img, layer1, layer2, override=None):
  """Attribution between spatial positions in two different layers."""

  # Set up a graph for doing attribution...
  with tf.Graph().as_default(), tf.Session(), gradient_override_map(override or {}):
    t_input = tf.placeholder_with_default(img, [None, None, None, 3])
    T = render.import_model(model, t_input, t_input)
    
    # Compute activations
    acts1 = T(layer1).eval()
    acts2 = T(layer2).eval({T(layer1) : acts1})
    
    # Construct gradient tensor
    # Backprop from spatial position (n_x, n_y) in layer2 to layer1.
    n_x, n_y = tf.placeholder("int32", []), tf.placeholder("int32", [])
    layer2_mags = tf.sqrt(tf.reduce_sum(T(layer2)**2, -1))[0]
    score = layer2_mags[n_x, n_y]
    t_grad = tf.gradients([score], [T(layer1)])[0]
    
    # Compute attribution backwards from each positin in layer2
    attrs = []
    for i in range(acts2.shape[1]):
      attrs_ = []
      for j in range(acts2.shape[2]):
        grad = t_grad.eval({n_x : i, n_y : j, T(layer1) : acts1})
        # linear approximation of imapct
        attr = np.sum(acts1 * grad, -1)[0]
        attrs_.append(attr)
      attrs.append(attrs_)
  return np.asarray(attrs)

def orange_blue(a,b,clip=False):
  if clip:
    a,b = np.maximum(a,0), np.maximum(b,0)
  arr = np.stack([a, (a + b)/2., b], -1)
  arr /= 1e-2 + np.abs(arr).max()/1.5
  arr += 0.3
  return arr

def spatial_spatial_attr(img, layer1, layer2, hint_label_1=None, hint_label_2=None, override=None):
  
  hint1 = orange_blue(
      raw_class_spatial_attr(img, layer1, hint_label_1, override=override),
      raw_class_spatial_attr(img, layer1, hint_label_2, override=override),
      clip=True
  )
  hint2 = orange_blue(
      raw_class_spatial_attr(img, layer2, hint_label_1, override=override),
      raw_class_spatial_attr(img, layer2, hint_label_2, override=override),
      clip=True
  )

  attrs = raw_spatial_spatial_attr(img, layer1, layer2, override=override)
  attrs = attrs / attrs.max()

  def blur(x, w1, w2):
  """Spatially blur a 4D tensor."""
  x_ = tf.pad(x, [(0,0), (1,1), (1,1), (0,0)], "CONSTANT")
  x_jitter_hv = (x_[:, 2:, 1:-1] + x_[:, :-2, 1:-1] + x_[:, 1:-1, 2:] + x_[:, 1:-1, :-2])/4.
  x_jitter_diag = (x_[:, 2:, 2:] + x_[:, 2:, :-2] + x_[:, :-2, 2:] + x_[:, :-2, :-2])/4.
  return (1-w1-w2)*x + w1*x_jitter_hv + w2*x_jitter_diag

def make_MaxSmoothPoolGrad(blur_hack=False):
  """Create a relaxed version of the MaxPool gradient.
  
  GoogLeNet's use of MaxPooling creates a lot of gradient artifacts. This
  function creates a fake gradient that gets rid of them, reducing distractions
  in our UI demos.
  
  Be very very careful about using this in real life. It hides model behavior
  from you. This can help you see other things more clearly, but in most cases
  you probably should do something else.
  
  We're actively researching what's going on here.
  
  Args:
    blur_hack: If True, use the second less principled trick of slightly
      blurring the gradient to get rid of checkerboard artifacts.
  
  Returns:
    Gradient function.
    
  """
  def MaxPoolGrad(op, grad):
    inp = op.inputs[0]
    
    # Hack 1 (moderately principled): use a relaxation of the MaxPool grad
    # ---------------------------------------------------------------------
    #
    # Construct a pooling function where, if we backprop through it,
    # gradients get allocated proportional to the input activation.
    # Then backpropr through that instead.
    #
    # In some ways, this is kind of spiritually similar to SmoothGrad
    # (Smilkov et al.). To see the connection, note that MaxPooling introduces
    # a pretty arbitrary discontinuity to your gradient; with the right
    # distribution of input noise to the MaxPool op, you'd probably smooth out
    # to this. It seems like this is one of the most natural ways to smooth.
    #
    # We'll probably talk about this and related things in future work.
    
    op_args = [op.get_attr("ksize"), op.get_attr("strides"), op.get_attr("padding")]
    smooth_out = tf.nn.avg_pool(inp**2, *op_args)/ (1e-2+tf.nn.avg_pool(tf.abs(inp), *op_args))
    inp_smooth_grad = tf.gradients(smooth_out, [inp], grad)[0]
    
    # Hack 2 (if argument is set; not very principled) 
    # -------------------------------------------------
    #
    # Slightly blur gradient to get rid of checkerboard artifacts.
    # Note, this really isn't principled. We're working around / hiding a bad
    # property of the model. It should really be fixed by better model design.
    #
    # We do this so that the artifacts don't distract from the UI demo, but we
    # don't endorse people doing it in real applications.
    
    if blur_hack:
      inp_smooth_grad = blur(inp_smooth_grad, 0.5, 0.25)
      
    return inp_smooth_grad
  return MaxPoolGrad


def compare_attrs(img, layer1, layer2, hint_label_1, hint_label_2):
  print "Normal gradient:\n"

  spatial_spatial_attr(img, layer1, layer2,
                       hint_label_1=hint_label_1, hint_label_2=hint_label_2)

  print "\nSmooth MaxPool Grad:"
  print "(note the subtle checkerboard patterns)\n"

  spatial_spatial_attr(img, layer1, layer2,
                       hint_label_1=hint_label_1, hint_label_2=hint_label_2,
                       override={"MaxPool": make_MaxSmoothPoolGrad()})

  print "\nSmooth + Blur MaxPool Grad:\n"

  spatial_spatial_attr(img, layer1, layer2,
                       hint_label_1=hint_label_1, hint_label_2=hint_label_2,
                       override={"MaxPool": make_MaxSmoothPoolGrad(blur_hack=True)})



imageData = np.load('playGames_Images.npy')

compare_attrs(np.float32(imageData[10:11,:]), "conv2d_4/Relu", "conv2d_5/Relu", "work", "play games")
