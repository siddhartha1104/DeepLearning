def run_experiment(m1, m2, m3, eps, N, seq_length):
    actions = [Actions(m1), Actions(m2), Actions(m3)]
    data = np.empty((N, seq_length, 224, 224, 3))  # Initialize data array for sequences of frames

    for i in range(N):
        frames = []
        for j in range(seq_length):
            p = np.random.random()
            if p < eps:
                k = np.random.choice(3)  # Explore
            else:
                k = np.argmax([a.mean for a in actions])  # Exploit
            x = actions[k].choose()  # Sample from the chosen action
            actions[k].update(x)  # Update the action's estimated mean

            # Generate a random frame (replace with actual frame generation code)
            frame = np.random.rand(224, 224, 3)
            frames.append(frame)

        data[i] = np.array(frames)

    return data