import torch

def torch_equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)#.type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255
        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]]) 
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)
        
        return result.type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    image = image.type(torch.float)
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 2)
    return image


def find_nearest_above(my_array, target):
    diff = my_array - target
    mask = diff <= -1
    # We need to mask the negative differences
    # since we are looking for values above
    if torch.all(mask):
        c = torch.abs(diff).argmin()
        return c # returns min index of the nearest if target is greater than any value
    masked_diff = diff.clone()
    masked_diff[mask] = 9999
    return masked_diff.argmin()


def hist_match(source, template):
    s = source.view(-1) 
    t = template.view(-1) 
    s_values, bin_idx, s_counts = torch.unique(s, return_inverse=True, return_counts=True) 
    t_values, t_counts = torch.unique(t, return_counts=True) 
    s_quantities = torch.cumsum(s_counts,0).type(torch.float)
    t_quantities = torch.cumsum(t_counts,0).type(torch.float)
    s_quantities/=s_quantities[s_quantities.shape[0]-1]
    t_quantities/=t_quantities[t_quantities.shape[0]-1]
    sour = (s_quantities * 255).type(torch.long) 
    temp = (t_quantities * 255).type(torch.long) 
    b = torch.zeros(sour.shape) 
    for i in range(sour.shape[0]):
        b[i] = find_nearest_above(temp, sour[i])

    s=b[bin_idx] 
    return s.view(source.shape)
    
def hist_match_dark_prior(img):
# input: img[h, w, c]
# output:res[h, w, c]
    result = img.clone()
    result = torch_equalize(result)
    dark_prior,_ = torch.min(result, axis=2)
    for i in range(3):
        result[:,:,i] = hist_match(result[:,:,i], dark_prior)
    return result
a=torch.tensor([1,2,2,3,3,3,4,4,4,4,5,5,5,5,5,6,6,6,6,6,6,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8])
a=a.reshape((3,4,3)).cuda()
b=hist_match_dark_prior(a)
print(a)
print(b)

