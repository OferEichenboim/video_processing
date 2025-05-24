import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "123456789"
ID2 = "987654321"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    #state_drifted = s_prior.copy()
    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""
    #drift according to prior
    s_prior_copy = s_prior.copy()
    s_prior_copy[0,:] = s_prior_copy[0,:] + s_prior_copy[4,:]
    s_prior_copy[1,:] = s_prior_copy[1,:] + s_prior_copy[5,:]

    
    s_prior_copy[0,:] = np.maximum(s_prior_copy[0,:],s_prior_copy[2,:])
    s_prior_copy[1,:] = np.maximum(s_prior_copy[1,:],s_prior_copy[3,:])

    state_drifted = s_prior_copy

    # Add noise to position (x, y)
    s_prior_copy[0, :] += np.random.randn(*s_prior_copy[0, :].shape) * 4.5
    s_prior_copy[1, :] += np.random.randn(*s_prior_copy[1, :].shape) * 2.0

    # Add noise to velocity (vx, vy)
    s_prior_copy[4, :] += np.random.randn(*s_prior_copy[4, :].shape) * 2.0
    s_prior_copy[5, :] += np.random.randn(*s_prior_copy[5, :].shape) * 0.75

    state_drifted[0:2, :] = np.round(state_drifted[0:2, :])
    state_drifted = state_drifted.astype(int)



    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16, 16, 16))
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    
    #get the subpotion of the image
    x1 = max(state[0] - state[2], 0)
    x2 = min(state[0] + state[2], image.shape[1])
    y1 = max(state[1] - state[3], 0)
    y2 = min(state[1] + state[3], image.shape[0])
    
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty crop region: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
    cropped = image[y1:y2, x1:x2, :]

    #quantization
    #cropped = cropped//16
    cropped_quant = np.floor(cropped.astype(float) * 15.0 / 255.0).astype(int)
    cropped_quant = np.clip(cropped_quant, 0, 15)

    #create histogram
    hist = np.zeros((16,16,16))
    for i in range(0,y2-y1):
        for j in range(0,x2-x1):
            hist[cropped_quant[i,j,0], cropped_quant[i,j,1], cropped_quant[i,j,2]]+=1

    total = np.sum(hist)
    if total == 0:
        raise ValueError("Histogram sum is zero — possible empty or black crop region.")

    # normalize and reshape
    hist = hist/np.sum(hist)
    hist = hist.reshape(-1,1)

    return hist



def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    
    for n in range(previous_state.shape[1]):
        r = np.random.random()
        for j in range(cdf.shape[0]):
            if cdf[j]>=r:
                S_next[:,n] = previous_state[:,j]
                #S_next[:2,n] +=S_next[4:,n]
                #S_next[4:,n] += np.array([0,0])
                break
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = 0
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    distance = np.exp(20 * np.sum(np.sqrt(np.multiply(p, q)))) 

    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    S_avg = np.floor(np.mean(state, axis=1))
    x_avg = S_avg[0] - S_avg[2] # x center - half width
    y_avg = S_avg[1] - S_avg[3] # y center - half height
    w_avg = 2 * S_avg[2] # half width * 2
    h_avg = 2 * S_avg[3] # half height * 2
    # Draw the average rectangle
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect) 
    #draw
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    MaxIndex = np.argmax(W)
    x_max = state[0, MaxIndex] - state[2, MaxIndex]  # x center - half width
    y_max = state[1, MaxIndex] - state[3, MaxIndex]  # y center - half height
    w_max = 2 * state[2, MaxIndex]  # half width * 2
    h_max = 2 * state[3, MaxIndex]  # half height * 2
    
    #draw
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    W = np.zeros(N)
    for i in range(N):
        # Compute the histogram for each particle
        p = compute_normalized_histogram(image, S[:, i])
        # Compute the Bhattacharyya distance
        W[i] = bhattacharyya_distance(p, q)
    # Normalize weights
    W = W / np.sum(W)
    # Compute CDF
    C = np.cumsum(W)


    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W = np.zeros(N)
        for i in range(N):
            # Compute the histogram for each particle
            p = compute_normalized_histogram(image, S[:, i])
            # Compute the Bhattacharyya distance
            W[i] = bhattacharyya_distance(p, q)
        # Normalize weights
        W = W / np.sum(W)
        # Compute CDF
        C = np.cumsum(W)


        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
