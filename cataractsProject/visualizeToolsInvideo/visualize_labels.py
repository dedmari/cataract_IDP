"""
TODO:
1) create 8 plots: 
    x axis is frame number per video
    y axis is for first plot the seven stages of the surgery
                  rest 7 plots either 0 or 1 depending on whether the tool X is in the frame or not
    -> use the extra/temp_labels labels (whole video)
    
2) create correlation between tools and phase (14 x 14 table)
    mat[i][j] = how many times is elem i (tool or phase) present with elem j (tool or phase) in a frame
        -> per video
        
3) also percentage of one/two/three/four/... occurring at the same time during the video
"""

import numpy as np
import matplotlib.pyplot as plt

phases = {"Preparation": 0,
          "CalotTriangleDissection": 1,
          "ClippingCutting": 2,
          "GallbladderDissection": 3,
          "GallbladderPackaging": 4,
          "CleaningCoagulation": 5,
          "GallbladderRetraction": 6}

phases_list = ["Preparation", "Calot triangle dissection", "Clipping & cutting", "Gallbladder dissection",
               "Gallbladder packaging", "Cleaning & coagulation", "Gallbladder retraction"]

tools = {0: 'Grasper',
         1: 'Bipolar',
         2: 'Hook',
         3: 'Scissors',
         4: 'Clipper',
         5: 'Irrigator',
         6: 'Specimen bag'}

tools_list = [tools[i] for i in sorted(list(tools.keys()))]


def construct_label_info_per_video(video_no, save_correlation=True, save_graphs=True):
    label_location = "./video-annotation/video" + "{0:0>2}".format(video_no) + ".txt"
    labels = {'nr_frames': 0, 'frame_labels': []}
    tool_labels = []
    phase_labels = []
    with open(label_location, "r") as f:
        for line_no, line in enumerate(f):
            if line_no:
                line = line.split()[1:]
                tool_label = np.array([int(x) for x in line[2:]])
                phase_label = np.zeros((7, ), dtype='int')
                phase_label[phases[line[1]]] = 1
                frame_label = {'frame': int(line[0]), 'tools': tool_label, 'phase': phase_label}
                tool_labels.append(tool_label)
                phase_labels.append(phase_label)
                labels['frame_labels'].append(frame_label)
    labels['nr_frames'] = len(labels['frame_labels'])
    tool_labels = np.array(tool_labels)
    phase_labels = np.array(phase_labels)
    # print(labels['nr_frames'])

    if save_correlation:
        correlation = np.zeros((14, 14), dtype='int')
        # titles = [''] + tools_list + phases_list
        titles = [''] + \
                 ["Tool "+str(i+1) for i in range(len(tools_list))] + \
                 ["Phase "+str(i+1) for i in range(len(phases_list))]
        col_widths = [len(x) for x in titles]
        col_widths[0] = max(col_widths)
        column_formats = [" >" + str(col_widths[i]) for i in range(len(titles))]

        output = ""
        output_line = ["" for _ in titles]
        nr_titles = len(titles)
        for row in range(nr_titles):
            if not row:
                output_line = [("{0:" + column_formats[i] + "}").format(titles[i]) + ("\t" if i == nr_titles//2 else "")
                               for i in range(nr_titles)]
            else:
                for col in range(nr_titles):
                    if not col:
                        out = titles[row]
                    else:
                        i = row - 1
                        j = col - 1
                        if i < 7:
                            if j < 7:
                                correlation[i, j] = np.sum(np.logical_and(tool_labels[:, i], tool_labels[:, j]))
                            else:
                                correlation[i, j] = np.sum(np.logical_and(tool_labels[:, i], phase_labels[:, j-7]))
                        else:
                            if j < 7:
                                correlation[i, j] = np.sum(np.logical_and(phase_labels[:, i-7], tool_labels[:, j]))
                            else:
                                correlation[i, j] = np.sum(np.logical_and(phase_labels[:, i-7], phase_labels[:, j-7]))
                        out = str(correlation[i, j])
                    output_line[col] = ("{0:" + column_formats[col] + "}").format(out) + ("\t" if col == nr_titles//2 else "")
            output += " ".join(output_line) + "\n" + ("\n" if row == nr_titles//2 else "")
        # print(correlation)
        filename = "./video-correlation/tool-phase-correlation-video{0:0>2}.txt".format(video_no)
        with open(filename, 'w') as f:
            f.write(output)

    if save_graphs:
        nr_simultaneous_tools = np.array([np.sum(np.count_nonzero(tool_labels, axis=1) == x) for x in range(0, 8)])
        num_plots = tool_labels.shape[1]
        x = np.arange(labels['nr_frames'])
        add_displacement = np.expand_dims(np.array([x for x in range(num_plots)]), axis=0)
        tool_labels += 5 * add_displacement
        phase_labels = np.multiply(phase_labels, add_displacement)

        figure = plt.figure(video_no, )
        plt.subplot(311)  # first subplot in figure video_no
        plt.title("Tool presence in video "+str(video_no))
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
        #legend = []
        for i in range(num_plots):
            plt.plot(x, tool_labels[:, i])
            #legend.append(tools[i])
        #plt.legend(legend, ncol=num_plots, loc='upper center', fancybox=True)
        plt.xlim(xmin=-10)
        plt.ylim(ymin=-1, ymax=5*tool_labels.shape[1])
        plt.yticks(5 * np.arange(tool_labels.shape[1]), tools_list)
        # play with axis => y in [-1; 36]

        plt.subplot(312)  # second subplot in figure video_no
        plt.title("Phases in video " + str(video_no))
        plt.plot(x, np.sum(phase_labels, axis=1))
        plt.xlim(xmin=-10)
        plt.ylim(ymin=-1, ymax=phase_labels.shape[1])
        plt.yticks(np.arange(phase_labels.shape[1]), phases_list)
        # play with axis => y in [-1; 8]

        plt.subplot(313)  # third subplot in figure video_no
        # plt.plot(np.arange(nr_simultaneous_tools.shape[0]), nr_simultaneous_tools, 'g|')
        plt.bar(np.arange(nr_simultaneous_tools.shape[0]) - 0.5, nr_simultaneous_tools, 1, color='g')
        plt.xlabel('number of simultaneous tools')
        plt.ylabel('number of frames')
        plt.title("Histogram of simultaneous tools in video "+str(video_no))
        plt.xlim(xmin=-1, xmax=nr_simultaneous_tools.shape[0])

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        # plt.show()
        figure.set_size_inches(w=23.01, h=12.32)
        figure.savefig("./video-plots/video" + "{0:0>2}".format(video_no) + ".png")
        plt.close(figure)

if __name__ == "__main__":
    starting_video = 1
    ending_video = 80
    for i in range(starting_video, ending_video + 1):
        construct_label_info_per_video(i, save_correlation=True, save_graphs=False)
    input("Press ENTER to exit program...")
