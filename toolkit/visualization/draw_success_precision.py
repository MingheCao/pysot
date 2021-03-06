import matplotlib.pyplot as plt
import numpy as np

from .draw_utils import COLOR, LINE_STYLE

def draw_success_precision(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name == bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        plt.plot(thresholds, np.mean(value, axis=0),
                color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    plt.show()


    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()

color_lines={'Ours(SiamRPN++)':{'color':COLOR[0],'line':LINE_STYLE[0]},
             'SiamRPN++':{'color':COLOR[1],'line':LINE_STYLE[1]},
             'Ours(SiamRPN)': {'color': COLOR[2], 'line': LINE_STYLE[0]},
             'SiamRPN': {'color': COLOR[3], 'line': LINE_STYLE[1]},
             'DaSiamRPN': {'color': COLOR[4], 'line': LINE_STYLE[0]},
             'UpdateNet': {'color': COLOR[5], 'line': LINE_STYLE[1]}}

def draw_success_precision_paper(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, save=False, save_path = None,axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('Overlap threshold')
    plt.ylabel('Success rate')
    if attr == 'ALL':
        plt.title(r'\textbf{Success plots of OPE on %s}' % (name))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name in bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]

        plt.plot(thresholds, np.mean(value, axis=0),
                color=color_lines[tracker_name]['color'], linestyle=color_lines[tracker_name]['line'],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    if save:
        plt.savefig(save_path + attr + '-success.jpg',bbox_inches='tight',pad_inches = 0,dpi=300)
    # plt.show()
    plt.clf()
    plt.close(fig)


    # if precision_ret:
    #     # norm precision plot
    #     fig, ax = plt.subplots()
    #     ax.grid(b=True)
    #     ax.set_aspect(50)
    #     plt.xlabel('Location error threshold')
    #     plt.ylabel('Precision')
    #     if attr == 'ALL':
    #         plt.title(r'\textbf{Precision plots of OPE on %s}' % (name))
    #     else:
    #         plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr))
    #     plt.axis([0, 50]+axis)
    #     precision = {}
    #     thresholds = np.arange(0, 51, 1)
    #     for tracker_name in precision_ret.keys():
    #         value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
    #         precision[tracker_name] = np.mean(value, axis=0)[20]
    #     for idx, (tracker_name, pre) in \
    #             enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
    #         if tracker_name in bold_name:
    #             label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
    #         else:
    #             label = "[%.3f] " % (pre) + tracker_name
    #         value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
    #         plt.plot(thresholds, np.mean(value, axis=0),
    #                 color=color_lines[tracker_name]['color'], linestyle=color_lines[tracker_name]['line'],label=label, linewidth=2)
    #     ax.legend(loc='lower right', labelspacing=0.2)
    #     ax.autoscale(enable=True, axis='both', tight=True)
    #     xmin, xmax, ymin, ymax = plt.axis()
    #     ax.autoscale(enable=False)
    #     ymax += 0.03
    #     plt.axis([xmin, xmax, ymin, ymax])
    #     plt.xticks(np.arange(xmin, xmax+0.01, 5))
    #     plt.yticks(np.arange(ymin, ymax, 0.1))
    #     ax.set_aspect((xmax - xmin)/(ymax-ymin))
    #     if save:
    #         plt.savefig( save_path + attr + '-precision.jpg', bbox_inches='tight',
    #                 pad_inches=0, dpi=300)
    #     # plt.show()
    #
    # # norm precision plot
    # if norm_precision_ret:
    #     fig, ax = plt.subplots()
    #     ax.grid(b=True)
    #     plt.xlabel('Location error threshold')
    #     plt.ylabel('Precision')
    #     if attr == 'ALL':
    #         plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
    #     else:
    #         plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
    #     norm_precision = {}
    #     thresholds = np.arange(0, 51, 1) / 100
    #     for tracker_name in precision_ret.keys():
    #         value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
    #         norm_precision[tracker_name] = np.mean(value, axis=0)[20]
    #     for idx, (tracker_name, pre) in \
    #             enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
    #         if tracker_name == bold_name:
    #             label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
    #         else:
    #             label = "[%.3f] " % (pre) + tracker_name
    #         value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
    #         plt.plot(thresholds, np.mean(value, axis=0),
    #                 color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
    #     ax.legend(loc='lower right', labelspacing=0.2)
    #     ax.autoscale(enable=True, axis='both', tight=True)
    #     xmin, xmax, ymin, ymax = plt.axis()
    #     ax.autoscale(enable=False)
    #     ymax += 0.03
    #     plt.axis([xmin, xmax, ymin, ymax])
    #     plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
    #     plt.yticks(np.arange(ymin, ymax, 0.1))
    #     ax.set_aspect((xmax - xmin)/(ymax-ymin))
    #     plt.show()

from matplotlib import font_manager

fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

color_lines_chn={'本文方法(SiamRPN++)':{'color':COLOR[0],'line':LINE_STYLE[0]},
             'SiamRPN++':{'color':COLOR[1],'line':LINE_STYLE[1]},
             '本文方法(SiamRPN)': {'color': COLOR[2], 'line': LINE_STYLE[0]},
             'SiamRPN': {'color': COLOR[3], 'line': LINE_STYLE[1]},
             'DaSiamRPN': {'color': COLOR[4], 'line': LINE_STYLE[0]},
             'UpdateNet': {'color': COLOR[5], 'line': LINE_STYLE[1]}}

def draw_success_precision_thesis(success_ret, name, videos, attr, precision_ret=None,
        norm_precision_ret=None, bold_name=None, save=False, save_path = None,axis=[0, 1]):
    # success plot
    fig, ax = plt.subplots()
    ax.grid(b=True)
    ax.set_aspect(1)
    plt.xlabel('重叠',fontproperties=fontP)
    plt.ylabel('bb',fontproperties=fontP)
    if attr == 'ALL':
        plt.title(r'\textbf{%s}' % (name))
    else:
        plt.title(r'\textbf{Success plots of OPE - %s}' % (attr))
    plt.axis([0, 1]+axis)
    success = {}
    thresholds = np.arange(0, 1.05, 0.05)
    for tracker_name in success_ret.keys():
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]
        success[tracker_name] = np.mean(value)
    for idx, (tracker_name, auc) in  \
            enumerate(sorted(success.items(), key=lambda x:x[1], reverse=True)):
        if tracker_name in bold_name:
            label = r"\textbf{[%.3f] %s}" % (auc, tracker_name)
        else:
            label = "[%.3f] " % (auc) + tracker_name
        value = [v for k, v in success_ret[tracker_name].items() if k in videos]

        plt.plot(thresholds, np.mean(value, axis=0),
                color=color_lines_chn[tracker_name]['color'], linestyle=color_lines_chn[tracker_name]['line'],label=label, linewidth=2)
    ax.legend(loc='lower left', labelspacing=0.2)
    ax.autoscale(enable=True, axis='both', tight=True)
    xmin, xmax, ymin, ymax = plt.axis()
    ax.autoscale(enable=False)
    ymax += 0.03
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xticks(np.arange(xmin, xmax+0.01, 0.1))
    plt.yticks(np.arange(ymin, ymax, 0.1))
    ax.set_aspect((xmax - xmin)/(ymax-ymin))
    if save:
        plt.savefig(save_path + attr + '-success.jpg',bbox_inches='tight',pad_inches = 0,dpi=300)
    # plt.show()


    if precision_ret:
        # norm precision plot
        fig, ax = plt.subplots()
        ax.grid(b=True)
        ax.set_aspect(50)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Precision plots of OPE - %s}' % (attr))
        plt.axis([0, 50]+axis)
        precision = {}
        thresholds = np.arange(0, 51, 1)
        for tracker_name in precision_ret.keys():
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name in bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=color_lines_chn[tracker_name]['color'], linestyle=color_lines_chn[tracker_name]['line'],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 5))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        if save:
            plt.savefig( save_path + attr + '-precision.jpg', bbox_inches='tight',
                    pad_inches=0, dpi=300)
        # plt.show()

    # norm precision plot
    if norm_precision_ret:
        fig, ax = plt.subplots()
        ax.grid(b=True)
        plt.xlabel('Location error threshold')
        plt.ylabel('Precision')
        if attr == 'ALL':
            plt.title(r'\textbf{Normalized Precision plots of OPE on %s}' % (name))
        else:
            plt.title(r'\textbf{Normalized Precision plots of OPE - %s}' % (attr))
        norm_precision = {}
        thresholds = np.arange(0, 51, 1) / 100
        for tracker_name in precision_ret.keys():
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            norm_precision[tracker_name] = np.mean(value, axis=0)[20]
        for idx, (tracker_name, pre) in \
                enumerate(sorted(norm_precision.items(), key=lambda x:x[1], reverse=True)):
            if tracker_name == bold_name:
                label = r"\textbf{[%.3f] %s}" % (pre, tracker_name)
            else:
                label = "[%.3f] " % (pre) + tracker_name
            value = [v for k, v in norm_precision_ret[tracker_name].items() if k in videos]
            plt.plot(thresholds, np.mean(value, axis=0),
                    color=COLOR[idx], linestyle=LINE_STYLE[idx],label=label, linewidth=2)
        ax.legend(loc='lower right', labelspacing=0.2)
        ax.autoscale(enable=True, axis='both', tight=True)
        xmin, xmax, ymin, ymax = plt.axis()
        ax.autoscale(enable=False)
        ymax += 0.03
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xticks(np.arange(xmin, xmax+0.01, 0.05))
        plt.yticks(np.arange(ymin, ymax, 0.1))
        ax.set_aspect((xmax - xmin)/(ymax-ymin))
        plt.show()
