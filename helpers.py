import torch

"""#Helper Functions"""

#the functions we'll use
#NOTE f is from POL inverse gradient

#puts the weights into a list
def weights_to_list(weights):
  with torch.no_grad():
    weights_list = []
    for weight in weights:
      list_t = weight.view(-1)
      for element in list_t:
        weights_list.append(element.item())

    return weights_list

#puts the weights into a list, but faster
def weights_to_list_fast(weights):
  with torch.no_grad():
    weights_list = []
    for weight in weights:
      list_t = weight.view(-1).tolist()
      weights_list = weights_list + list_t

    return weights_list

#sets weights in a model given a 1d list,
#and list containg the parameters of a model (obtained through name,parameter in model.named_parameters())
def set_weights(x, weights):
  with torch.no_grad():
    start = 0
    #index = 0
    for weight in weights:
      length = len(weight.view(-1))
      array = x[start:start+length]
      weight_new = torch.Tensor(array).view(*weight.shape)

      dimensions = len(weight.shape)
      if dimensions == 2:
        for i in range(weight.shape[0]):
          for j in range(weight.shape[1]):
            weight[i][j] = weight_new[i][j]

      elif dimensions == 1:
        for i in range(weight.shape[0]):
          weight[i] = weight_new[i]

      #index +=1
      start += length


# set weight like above example, but faster
def set_weights_fast(x, weights):
  with torch.no_grad():
    start = 0
    #index = 0
    for weight in weights:
      length = len(weight.view(-1))
      array = x[start:start+length]
      weight_new = torch.Tensor(array).view(*weight.shape)

      weight.data.copy_(weight_new)
      #index +=1
      start += length


def validate(model,dataloader):
  total = 0
  correct = 0

  for img,label in dataloader:
    img = img.to(device)
    label = label.to(device)
    out_probs = model(img)
    #print("out shape", out_probs.shape)
    out = torch.argmax(out_probs, dim=1)


    label = label.view(out.shape)

    total+= len(img)
    correct += torch.sum(out == label)

    #print("got here")

  return total, correct


