import matplotlib.pyplot as plt


def make_plots(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Model loss.png")
    plt.show()


# ## Gender model ##
# fig, (ax1, ax2) = plt.subplots(1, 2)

# # fig.suptitle('Horizontally stacked subplots')
# ax1.plot(gender_history.history['loss'])
# ax1.plot(gender_history.history['val_loss'])
# ax1.set_title('model loss')
# ax1.set(xlabel='epoch', ylabel='loss')
# ax1.legend(['train', 'val'], loc='upper left')
# # ax1.show()

# #print(gender_history.history.keys())

# ax2.plot(gender_history.history['accuracy'])
# ax2.plot(gender_history.history['val_accuracy'])
# ax2.set_title('model accuracy')
# ax1.set(xlabel='epoch', ylabel='accuracy')
# ax2.legend(['train', 'val'], loc='upper left')
