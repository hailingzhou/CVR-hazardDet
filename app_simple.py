import numpy as np
import streamlit as st
from PIL import Image
# import cv2

import torch
import os

from inference import inference

st.set_option('deprecation.showfileUploaderEncoding', False)


if 'scene' not in st.session_state:
    st.session_state.scene = False

if 'flag' not in st.session_state:
    st.session_state.flag = False

if 'custom_first_word' not in st.session_state:
    st.session_state.custom_first_word = False

if 'custom_second_word' not in st.session_state:
    st.session_state.custom_second_word = False

if 'custom_third_word' not in st.session_state:
    st.session_state.custom_third_word = False


# st.write(st.session_state)
st.markdown('<font size=10><center>__Hazard Detection__</font></center>', unsafe_allow_html=True)


st.write("")

file_list = []
flag = None


st.markdown('rule_set1', unsafe_allow_html=True)
rule_set1 = st.text_area('', '''
No person;
No person smoking or holding cigarette;
No person wearing or holding helmet;
''', height= 200)

# st.write(rule_set1.split(';')[:-1])


st.markdown('rule_set2', unsafe_allow_html=True)
rule_set2 = st.text_area('','''
No fire;
No person climbing on or close to ladder;
No person carrying or close to goods;
No person wearing or holding safety rope;
''', height= 200)


# if st.session_state.scene == False:
#     st.markdown('<center><font color="#dd0000">Please choose one scene.</font><br /></center>', unsafe_allow_html=True)

#     # with st.sidebar:

#         # file_up = st.file_uploader("Upload an image", type="jpg", accept_multiple_files=True)
#         # if file_up is None:
#         #     st.stop()

#         # for file in file_up:
#         #     file_list.append(file.name)

#     st.write("")
#     st.write("")
#     col1, col2, col3= st.columns(3)


#     if col1.button("Indoor Scenes"):
#         st.session_state.scene = '1'

#     elif col2.button("Outdoor Scenes") or st.session_state.scene=='2':
#         st.session_state.scene = '2'

#     elif col3.button("Random Scenes") or st.session_state.scene=='3':
#         st.session_state.scene = '3'


#     # st.markdown('rule_set1', unsafe_allow_html=True)
#     # rule_set1 = st.text_area('', '''
#     # No person
#     # No person wearing or holding helmet
#     # No person standing on or sitting on ground
#     # No person close to or sitting on machine
#     # No person standing on or climbing on construction vehicles
#     # ''', height= 200)

#     # st.markdown('rule_set2', unsafe_allow_html=True)
#     # rule_set2 = st.text_area('','''
#     # No person smoking or holding cigarette
#     # No person climbing on or close to ladder
#     # No person driving or climbing on construction vehicles
#     # No person carrying or close to goods
#     # No person wearing or holding safety rope
#     # ''', height= 200)

# else:
#     st.markdown('<font color="#dd0000">You choose to use Scene {}</font>'.format(st.session_state.scene), unsafe_allow_html=True)
#     st.markdown('<font color="#dd0000">Total num of images.: 4</font>', unsafe_allow_html=True)

st.markdown('<center><font color="#dd0000">Please choose one scene.</font><br /></center>', unsafe_allow_html=True)

# with st.sidebar:

    # file_up = st.file_uploader("Upload an image", type="jpg", accept_multiple_files=True)
    # if file_up is None:
    #     st.stop()

    # for file in file_up:
    #     file_list.append(file.name)

st.write("")
st.write("")
col1, col2, col3= st.columns(3)


if st.session_state.scene == False:
    if col1.button("Indoor Scenes"):
        st.session_state.scene = '1'

    elif col2.button("Outdoor Scenes") or st.session_state.scene=='2':
        st.session_state.scene = '2'

    elif col3.button("Random Scenes") or st.session_state.scene=='3':
        st.session_state.scene = '3'








if st.session_state.scene == '1':


    files = os.listdir('D:\学习\数据集标注\demo\images\场景1')
    for file in files:
        file_list.append(os.path.join('D:\学习\数据集标注\demo\images\场景1', file))

    
    st.markdown('<font color="#dd0000">You choose to use Indoor Scenes</font>'.format(st.session_state.scene), unsafe_allow_html=True)
    st.markdown('<font color="#dd0000">Total num of images.: {}</font>'.format(len(file_list)), unsafe_allow_html=True)
    


    mode_col1, mode_col2, mode_col3= st.columns(3)

    if mode_col1.button("Use Rule_Set1 As an Example"):
        st.session_state.flag = "1"
        st.markdown('<font color="#dd0000">Using Rule_Set1 as an Example</font>', unsafe_allow_html=True)

    if mode_col2.button("Use Rule_Set2 As an Example"):
        st.session_state.flag = "2"
        st.markdown('<font color="#dd0000">Using Rule_Set2 as an Example</font>', unsafe_allow_html=True)

    if mode_col3.button("Use customized Rule_set"):
        st.session_state.flag = "3"
        st.markdown('<font color="#dd0000">Using Customized Rule_set as an Example</font>', unsafe_allow_html=True)
        st.session_state.scene = False


elif st.session_state.scene == '2':
    files = os.listdir('D:\学习\数据集标注\demo\images\场景2')
    for file in files:
        file_list.append(os.path.join('D:\学习\数据集标注\demo\images\场景2', file))

    st.markdown('<font color="#dd0000">You choose to use Outdoor Scenes</font>'.format(st.session_state.scene), unsafe_allow_html=True)
    st.markdown('<font color="#dd0000">Total num of images.: {}</font>'.format(len(file_list)), unsafe_allow_html=True)


    if st.button("Use Rule_Set1 As an Example"):
        st.session_state.flag = "1"

    if st.button("Use Rule_Set2 As an Example"):
        st.session_state.flag = "2"

    if st.button("Use customized Rule_set"):
        st.session_state.flag = "3"
        st.session_state.scene = False



elif st.session_state.scene == '3':
    files = os.listdir('D:\学习\数据集标注\demo\images\场景3')
    for file in files:
        file_list.append(os.path.join('D:\学习\数据集标注\demo\images\场景3', file))

    st.markdown('<font color="#dd0000">You choose to use Random Scenes</font>'.format(st.session_state.scene), unsafe_allow_html=True)
    st.markdown('<font color="#dd0000">Total num of images.: {}</font>'.format(len(file_list)), unsafe_allow_html=True)

    if st.button("Use Rule_Set1 As an Example"):
        st.session_state.flag = "1"

    if st.button("Use Rule_Set2 As an Example"):
        st.session_state.flag = "2"

    if st.button("Use customized Rule_set"):
        st.session_state.flag = "3"
        st.session_state.scene = False




if st.session_state.flag == "1":

    st.write('You are using Rule_Set1!!!!!')
    for file in file_list:
        # col1, col2 = st.columns(2)
        # col1.header("Original" + file)
        # col1.image(original, use_column_width=False)
        # col2.header("Segmentation")
        # col2.image(grayscale, use_column_width=False)
        # 

        rule_set = []
        for rule in rule_set1.split(';')[:-1]:
            rule_set.append(rule.lstrip())
        rule_violated_list = []
        st.markdown('<center>{}</center>'.format(file.split('\\')[-1]), unsafe_allow_html=True)
        # st.write(file)
        col1, col2, col3 = st.columns(3)
        image = Image.open(file)
        image = image.resize((256, 256))
        # st.image(image)
        col2.image(image, use_column_width=False)

        for rule in rule_set:
            result = inference(file.split('\\')[-1], rule)
            if result == 'yes':
                rule_violated_list.append(rule)
            
        


        st.markdown('Prediction Results of {}'.format(file.split('\\')[-1]), unsafe_allow_html=True)
        if len(rule_violated_list) != 0:
            st.markdown('<font color="#dd0000">There is a hazard!</font>'.format(file.split('\\')[-1]), unsafe_allow_html=True)
            st.markdown('rule violated:', unsafe_allow_html=True)
            for rule_violated in rule_violated_list:
                st.markdown('<font color="#dd0000">{}</font>'.format(rule_violated), unsafe_allow_html=True)
        else:
            st.markdown('<font color="#009900">There is no hazard!</font>'.format(file.split('\\')[-1]), unsafe_allow_html=True)



        

elif st.session_state.flag == '2':
    st.write('You are using Rule_Set2!!!!!')

    for file in file_list:
        # col1, col2 = st.columns(2)
        # col1.header("Original" + file)
        # col1.image(original, use_column_width=False)
        # col2.header("Segmentation")
        # col2.image(grayscale, use_column_width=False)
        # 

        rule_set = []
        for rule in rule_set2.split(';')[:-1]:
            rule_set.append(rule.lstrip())
        rule_violated_list = []
        st.markdown('<center>{}</center>'.format(file.split('\\')[-1]), unsafe_allow_html=True)
        # st.write(file)
        col1, col2, col3 = st.columns(3)
        image = Image.open(file)
        image = image.resize((256, 256))
        # st.image(image)
        col2.image(image, use_column_width=False)

        for rule in rule_set:
            result = inference(file.split('\\')[-1], rule)
            if result == 'yes':
                rule_violated_list.append(rule)
            
        


        st.markdown('Prediction Results of {}'.format(file.split('\\')[-1]), unsafe_allow_html=True)
        if len(rule_violated_list) != 0:
            st.markdown('<font color="#dd0000">There is a hazard!</font>'.format(file.split('\\')[-1]), unsafe_allow_html=True)
            st.markdown('rule violated:', unsafe_allow_html=True)
            for rule_violated in rule_violated_list:
                st.markdown('<font color="#dd0000">{}</font>'.format(rule_violated), unsafe_allow_html=True)
        else:
            st.markdown('<font color="#009900">There is no hazard!</font>'.format(file.split('\\')[-1]), unsafe_allow_html=True)


elif st.session_state.flag == '3':
    st.write('You are using Customized rules!!!!!')
























































# if st.button("Use Rule_Set1 As an Example"):
#     flag = "1"

# if st.button("Use Rule_Set2 As an Example"):
#     flag = "2"

# if st.button("Use customized Rule_set"):
#     flag = "3"


# file_select = st.multiselect('please choose one image', (file_list))

# # st.write(file_list)
# path = r'D:/学习/数据集标注/demo/images/'
# # with open("style.css") as f:
# #     st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

# col1, col2= st.columns(2)

# col1.header('Rule_set1')



# col2.header('Rule_set2')

# rule_set2 = col2.text_area('','''
#     It was the best of times, it was the worst of times, it was
#     the age of wisdom, it was the age of foolishness, it was
#     the epoch was the epoch of incredulity, it
#     was the season of Light, it was the season of Darkness, it
#     was the spring of hope, it was the winter of despair, (...)
#     ''', height= 100)


# st.markdown('<center><font color="#dd0000">Please follow the rules template.</font><br /></center>', unsafe_allow_html=True)


    

#     # st.image()

# st.header('Customized rule set')

# st.markdown("""
# <style>
# .big-font {
#     font-size:30px !important;
# }
# </style>
# """, unsafe_allow_html=True)


# col1, empty1, col2, empty2, col3 = st.columns([0.3, 0.3, 1.2, 0.3, 1.2])
# col1.write('')
# col1.write('')
# # col1.write('No')
# col1.markdown('<p class="big-font">No</p>', unsafe_allow_html=True)
# col3.write('')
# col3.write('')
# col3.markdown('<p class="big-font">one the surface</p>', unsafe_allow_html=True)
# col3.write('')
# with col2:
#     option = st.selectbox(
#         "choose one word",
#         ("Email", "Home phone", "Mobile phone"),
#         # label_visibility="hidden",
#         # disabled=st.session_state.disabled,
#     )


# html_str_1 = f"""
# <style>
# p.a {{
# font: bold 20px Courier;
# }}
# </style>
# <p class="a">No {option} on the surface</p>
# """

# html_str_2 = f"""
# <style>
# p.a {{
# font: bold 100px Courier;
# }}
# </style>
# <p class="a">You are using the following rule!</p>
# """
# st.markdown('<p class="big-font">You are using the following rule!</p>', unsafe_allow_html=True)
# st.markdown(html_str_1, unsafe_allow_html=True)


# import streamlit as st
# empty1,content1,empty2,content2,empty3=st.columns([0.3,1.2,0.3,1.2,0.3])
# with empty1:
#         st.empty()
# with content1:
#         st.write("here is the first column of content.")
# with empty2:
#         st.empty()
# with content2:
#         st.write("here is the second column of content.")
# with empty3:
#         st.empty()

# col1, col2, col3 = st.columns(3)

# state.inputs = state.inputs or set()

# c1, c2 = st.beta_columns([2, 1])

# input_string = c1.text_input("Input")
# state.inputs.add(input_string)

# # Get the last index of state.inputs if it's not empty
# last_index = len(state.inputs) - 1 if state.inputs else None

# # Automatically select the last input with last_index
# c2.selectbox("Input presets", options=list(state.inputs), index=last_index)





# if flag == "1":
#     st.write('You are using Rule_Set1!!!!!')
#     for file in file_select:
#         # col1, col2 = st.columns(2)
#         # col1.header("Original" + file)
#         # col1.image(original, use_column_width=False)
#         # col2.header("Segmentation")
#         # col2.image(grayscale, use_column_width=False)
#         # 
#         st.markdown('<center>{}</center>'.format(file), unsafe_allow_html=True)
#         # st.write(file)
#         col1, col2, col3 = st.columns(3)
#         image = Image.open(path + file)
#         image = image.resize((256, 256))
#         # st.image(image)
#         col2.image(image, use_column_width=False)

#         st.text_input('Prediction Results of {}'.format(file), 'yes')
# elif flag == '2':
#     st.write('You are using Rule_Set2!!!!!')
#     for file in file_select:
#         # col1, col2 = st.columns(2)
#         # col1.header("Original" + file)
#         # col1.image(original, use_column_width=False)
#         # col2.header("Segmentation")
#         # col2.image(grayscale, use_column_width=False)
#         # 
#         st.markdown('<center>{}</center>'.format(file), unsafe_allow_html=True)
#         # st.write(file)
#         col1, col2, col3 = st.columns(3)
#         image = Image.open(path + file)
#         image = image.resize((256, 256))
#         # st.image(image)
#         col2.image(image, use_column_width=False)

#         st.text_input('Prediction Results of {}'.format(file), 'yes')
# elif flag == '3':
    # title = st.write('Enter your rules!!', '')


    # col1, empty1, col2, empty2, col3 = st.columns([0.3, 0.3, 1.2, 0.3, 1.2])
    # col1.write('')
    # col1.write('')
    # # col1.write('No')
    # col1.markdown('<p class="big-font">No</p>', unsafe_allow_html=True)
    # col3.write('')
    # col3.write('')
    # col3.markdown('<p class="big-font">one the surface</p>', unsafe_allow_html=True)
    # col3.write('')
    # with col2:
    #     option = st.selectbox(
    #         "choose one word",
    #         ("Email", "Home phone", "Mobile phone"),
    #         # label_visibility="hidden",
    #         # disabled=st.session_state.disabled,
    #     )


    # html_str_1 = f"""
    # <style>
    # p.a {{
    # font: bold 20px Courier;
    # }}
    # </style>
    # <p class="a">No {option} on the surface</p>
    # """

    # html_str_2 = f"""
    # <style>
    # p.a {{
    # font: bold 100px Courier;
    # }}
    # </style>
    # <p class="a">You are using the following rule!</p>
    # """
    # st.markdown('<p class="big-font">You are using the following rule!</p>', unsafe_allow_html=True)
    # st.markdown(html_str_1, unsafe_allow_html=True)


    # for file in file_select:
    #     # col1, col2 = st.columns(2)
    #     # col1.header("Original" + file)
    #     # col1.image(original, use_column_width=False)
    #     # col2.header("Segmentation")
    #     # col2.image(grayscale, use_column_width=False)
    #     # 
    #     st.markdown('<center>{}</center>'.format(file), unsafe_allow_html=True)
    #     # st.write(file)
    #     col1, col2, col3 = st.columns(3)
    #     image = Image.open(path + file)
    #     image = image.resize((256, 256))
    #     # st.image(image)
    #     col2.image(image, use_column_width=False)

    #     st.text_input('Prediction Results of {}'.format(file), 'yes')










# if flag == "1":
#     from swin_unet_pre import SwinUnet, SwinUnet_config

#     config = SwinUnet_config()
#     model = SwinUnet(config, img_size=224, num_classes=5).cpu()
#     save_path = os.path.join('/dataset/VQA/BLIP', "best_model_seg.pth")
#     model.load_state_dict(torch.load(save_path))


# for file in file_select:
#     original, grayscale = inference(model, file) \
#         # / 255 #original.convert('LA')
#     col1, col2 = st.columns(2)
#     col1.header("Original" + file)
#     col1.image(original, use_column_width=False)
#     col2.header("Segmentation")
#     col2.image(grayscale, use_column_width=False)

# if flag == "2":
#     st.write("Not implemented yet!")


# ##################2D plot#####################################

# import numpy as np
# import nibabel as nib
# from ipywidgets import interact, interactive, IntSlider, ToggleButtons
# import matplotlib.pyplot as plt
# import seaborn as sns

# image_path = "./0_pred.nii.gz"
# image_path_2 = "./imagesTr/img0001.nii.gz"
# image_obj = nib.load(image_path)
# print(f'Type of the image {type(image_obj)}')

# image_data = image_obj.get_fdata()
# type(image_data)

# height, width, depth = image_data.shape
# st.write(f"The image object height: {height}, width:{width}, depth:{depth}")

# st.write(f'image value range: [{image_data.min()}, {image_data.max()}]')

# st.write(image_obj.header.keys())

# pixdim =  image_obj.header['pixdim']
# st.write(f'z轴分辨率： {pixdim[3]}')
# st.write(f'in plane 分辨率： {pixdim[1]} * {pixdim[2]}')


# depth_selection = st.slider('depth:',
#                           min_value=0,
#                           max_value=depth-1,
#                           value=(0, depth-1))

# height_selection = st.slider('height:',
#                           min_value=0,
#                           max_value=height-1,
#                           value=(0, height-1))
# width_selection = st.slider('width:',
#                           min_value=0,
#                           max_value=width-1,
#                           value=(0, width-1))
# st.write(depth_selection)
# st.write(depth_selection[-1])
# Define a channel to look at
# fig = plt.figure()
# plt.imshow(image_data[:, :, depth_selection[-1]], cmap='PuRd')
# plt.axis('off')
# st.pyplot(fig)

# fig = plt.figure()
# plt.imshow(image_data[height_selection[-1], :, :], cmap='PuRd')
# plt.axis('off')
# st.pyplot(fig)

# fig = plt.figure()
# plt.imshow(image_data[:, width_selection[-1], :], cmap='PuRd')
# plt.axis('off')
# st.pyplot(fig)

# """From https://matplotlib.org/3.1.0/gallery/mplot3d/voxels.html in Streamlit"""
# import matplotlib.pyplot as plt
# import numpy as np
# import streamlit as st

# # This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


# @st.cache
# def generate_data():
#     """Let's put the data in cache so it doesn't reload each time we rerun the script when modifying the slider"""
#     # prepare some coordinates
#     x, y, z = np.indices((8, 8, 8))

#     # draw cuboids in the top left and bottom right corners, and a link between them
#     cube1 = (x < 3) & (y < 3) & (z < 3)
#     cube2 = (x >= 5) & (y >= 5) & (z >= 5)
#     link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

#     # combine the objects into a single boolean array
#     voxels = cube1 | cube2 | link

#     colors = np.empty(voxels.shape, dtype=object)
#     colors[link] = 'red'
#     colors[cube1] = 'blue'
#     colors[cube2] = 'green'

#     return voxels, colors

# voxels, colors = generate_data()

# let's put sliders to modify view init, each time you move that the script is rerun, but voxels are not regenerated
# TODO : not sure that's the most optimized way to rotate axis but well, demo purpose
# azim = st.sidebar.slider("azim", 0, 90, 30, 1)
# elev = st.sidebar.slider("elev", 0, 360, 240, 1)

# and plot everything
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(voxels, facecolors=colors, edgecolor='k')
# ax.view_init(azim, elev)



#######################################################3D plot##############################################################

# import plotly.graph_objects as go
# import numpy as np
# import numpy as np
# import nibabel as nib
# from ipywidgets import interact, interactive, IntSlider, ToggleButtons
# import matplotlib.pyplot as plt
# import seaborn as sns

# # image_path = "./0_pred.nii.gz"
# # image_path = "./amos_0123.nii.gz"
# image_path = "./amos_0123_small_8x.nii.gz"
# image_obj = nib.load(image_path)
# image_data = image_obj.get_fdata()

# # image_data_small = image_data[::2, ::2, ::2]

# # del image_data
# image_data = image_data / 15

# values = image_data
# st.write(image_data.max())
# st.write(np.unique(image_data))
# height, width, depth = image_data.shape
# st.write(height) #64 #512
# st.write(width) #256 #512
# st.write(depth) #256 #112
# # X, Y, Z = np.mgrid[0:height:64j, 0:width:256j, 0:depth:256j]
# # X, Y, Z = np.mgrid[0:256:256j, 0:256:256j, 0:56:56j]
# # X, Y, Z = np.mgrid[0:height:32j, 0:width:128j, 0:depth:128j]
# X, Y, Z = np.mgrid[0:height:64j, 0:width:64j, 0:depth:14j]
# # X, Y, Z = np.mgrid[0:height:128j, 0:width:128j, 0:depth:28]
# # X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
# st.write('X')
# st.write(X.shape)
# st.write('Y')
# st.write(Y.shape)
# st.write('Z')
# st.write(Z.shape)
# # values = np.sin(X*Y*Z) / (X*Y*Z)
# # values = torch.zeros(40, 40, 40) + 0.5
# # st.write(values.shape)
# # values = image_data
# # list = []


# # st.write(list)
# st.write(values)
# st.write(values.max())
# st.write(np.unique(values))
# st.write(values.min())
# st.write(values.shape)

# # st.write(image_data.shape)
# n_colors = 15
# import plotly.express as px
# colors = px.colors.sample_colorscale("turbo", [n/(n_colors -1) for n in range(n_colors)])

# fig = go.Figure(data=go.Volume(
#     x=X.flatten(),
#     y=Y.flatten(),
#     z=Z.flatten(),
#     value=values.flatten(),
#     isomin=0.1,
#     isomax=0.8,
#     opacity=0.1, # needs to be small to see through all surfaces
#     surface_count=30, # needs to be a large number for good volume rendering
#     ))
# # ,nticks=10, range=[0,X]

# fig.update_layout(
#     scene=dict(
#         # xaxis=dict(tickmode='auto',autorange=False,),
#         # yaxis=dict(tickmode='auto',autorange=False,),
#         # zaxis=dict(tickmode='auto',autorange=False,),
#         aspectratio=dict(x=1, y=1, z=1)
#         ))

# fig.show()
# st.pyplot()












