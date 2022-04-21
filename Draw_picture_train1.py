import numpy as np
import matplotlib.pyplot as plt

'''
epoch = 15  ;  batch_size = 8


VGG11:
Total params: 128,851,122
Trainable params: 128,829,214
Non-trainable params: 21,908


VGG13:
Total params: 129,036,402
Trainable params: 129,014,110
Non-trainable params: 22,292


RESNET:
Total params: 11,718,082
Trainable params: 11,708,482
Non-trainable params: 9,600


MOBILE:
Total params: 4,274,818
Trainable params: 4,252,930
Non-trainable params: 21,888

'''


VGG11_loss = [2.2380411624908447, 1.8591996431350708, 1.636421799659729, 1.46615469455719, 1.2450194358825684,
              1.081541657447815, 0.9675431251525879, 0.8850924968719482, 0.7603005766868591, 0.6794772148132324,
              0.6778161525726318, 0.5614506602287292, 0.5169898271560669, 0.5023771524429321, 0.46225079894065857]
VGG11_acc = [0.1544772833585739, 0.3080308735370636, 0.41018256545066833, 0.4828298091888428, 0.5723212361335754,
             0.637361466884613, 0.6778417825698853, 0.7102260589599609, 0.7576614022254944, 0.7886329293251038,
             0.787328839302063, 0.8288415670394897, 0.8422625660896301, 0.8472071290016174, 0.864214301109314]
VGG11_val_loss = [9.06424331665039, 5.8763651847839355, 1.2143354415893555, 1.0023937225341797, 1.1724404096603394,
             0.6317670941352844, 0.6086130738258362, 0.3427242934703827, 0.6927711963653564, 0.30901482701301575,
             0.8948600888252258, 0.3996342122554779, 0.24765494465827942, 1.0899726152420044, 0.8310049176216125]
VGG11_val_acc = [0.09950000047683716, 0.2705000042915344, 0.5870000123977661, 0.6754999756813049, 0.6524999737739563,
                 0.8259999752044678, 0.8479999899864197, 0.8980000019073486, 0.9229999780654907, 0.9110000133514404,
                 0.9355000257492065, 0.9570000171661377, 0.921999990940094, 0.8755000233650208, 0.8949999809265137]


VGG13_loss = [2.278942346572876, 1.9625835418701172, 1.8300265073776245, 1.6894865036010742, 1.5386143922805786,
          1.4262021780014038, 1.3608652353286743, 1.2870392799377441, 1.2181349992752075, 1.2039631605148315,
          1.1135164499282837, 1.1102650165557861, 1.1127939224243164, 1.0640649795532227, 1.0695258378982544]
VGG13_acc =[0.16393175721168518, 0.30699849128723145, 0.36932188272476196, 0.412899374961853, 0.4558791518211365,
            0.49418604373931885, 0.5090197920799255, 0.5298848152160645, 0.5488480925559998, 0.5469462871551514,
            0.5737882852554321, 0.5727559328079224, 0.5714518427848816, 0.5794935822486877, 0.5829167366027832]
VGG13_val_loss =  [1.9345343112945557, 7.77593994140625, 1.3004920482635498, 1.4384657144546509, 1.0948574542999268,
                   1.2683806419372559, 1.4774304628372192, 1.157472014427185, 1.0531833171844482, 1.2117947340011597,
                   2.2785701751708984, 7.026256561279297, 1.6254255771636963, 1.0046300888061523, 9.717437744140625]
VGG13_val_acc =  [0.304500013589859, 0.18700000643730164, 0.5264999866485596, 0.46050000190734863, 0.5615000128746033,
                  0.5634999871253967, 0.5805000066757202, 0.5849999785423279, 0.5709999799728394, 0.5855000019073486,
                  0.5889999866485596, 0.4595000147819519, 0.5839999914169312, 0.5835000276565552, 0.3815000057220459]

Resnet18_loss = [1.7949753999710083, 0.7792201042175293, 0.49135541915893555, 0.3605203926563263, 0.2940848767757416,
                 0.2450653463602066, 0.2021605372428894, 0.18051640689373016, 0.16117839515209198, 0.14149008691310883,
                 0.12867385149002075, 0.12234838306903839, 0.1143534854054451, 0.10048246383666992, 0.09408318251371384]
Resnet18_acc = [0.3697022497653961, 0.739132821559906, 0.8424798846244812, 0.8881765007972717, 0.9126819968223572,
                0.9227885007858276, 0.9397413730621338, 0.9455553293228149, 0.9528363347053528, 0.958867609500885,
                0.9631058573722839, 0.9640295505523682, 0.9668006896972656, 0.9694631695747375, 0.973538339138031]
Resnet18_val_loss = [1.1367957592010498, 0.5813484787940979, 0.20631298422813416, 0.20205935835838318, 0.32862588763237,
                     0.14643147587776184, 0.25992077589035034, 0.25083673000335693, 0.15567484498023987, 0.12450160086154938,
                     0.23945370316505432, 0.14195197820663452, 0.07130333036184311, 0.046542610973119736, 0.05468910187482834]
Resnet18_val_acc = [0.6545000076293945, 0.8190000057220459, 0.9399999976158142, 0.9424999952316284, 0.8899999856948853,
                    0.9549999833106995, 0.9120000004768372, 0.9200000166893005, 0.9524999856948853, 0.968999981880188,
                    0.921500027179718, 0.9595000147819519, 0.9760000109672546, 0.9879999756813049, 0.9869999885559082]

Mobilenet_loss =[2.2259373664855957, 1.6086574792861938, 1.0530022382736206, 0.6309919357299805, 0.43831947445869446,
                 0.3411950170993805, 0.28756803274154663, 0.2542992830276489, 0.2158355414867401, 0.19428622722625732,
                 0.16818970441818237, 0.1611613929271698, 0.1423468440771103, 0.13271383941173553, 0.13170631229877472]
Mobilenet_acc = [0.141273632645607, 0.3756248652935028, 0.6091610789299011, 0.7957509160041809, 0.8713866472244263,
                 0.9036079049110413, 0.9207237362861633, 0.9301238656044006, 0.9397956728935242, 0.9455009698867798,
                 0.9533796906471252, 0.9552271366119385, 0.9615844488143921, 0.9638665318489075, 0.9640839099884033]
Mobilenet_val_loss = [1.9625240564346313, 1.2525173425674438, 0.9291898608207703, 0.315512090921402, 0.29343798756599426,
                      0.16879203915596008, 0.259146511554718, 0.19463686645030975, 0.37787485122680664, 0.5839920043945312,
                      0.1414400041103363, 0.20124086737632751, 0.10797417163848877, 0.12709416449069977, 0.054925791919231415]
Mobilenet_val_acc = [0.21050000190734863, 0.4869999885559082, 0.7164999842643738, 0.8985000252723694, 0.9004999995231628,
                     0.9490000009536743, 0.9279999732971191, 0.9470000267028809, 0.8930000066757202, 0.8399999737739563,
                     0.9605000019073486, 0.9465000033378601, 0.9729999899864197, 0.9714999794960022, 0.9865000247955322]

N = 15 # N=epochs
'''
1.前三个网络的loss
'''
plt.plot(np.arange(1, N+1), VGG11_loss, label="VGG11_loss(middle)")
plt.plot(np.arange(1, N+1), VGG13_loss, label="VGG13_loss(up)")
plt.plot(np.arange(1, N+1), Resnet18_loss, label="ResNet18_loss(dowm)")


plt.annotate('VGG11', xy=(6, 1.05), xytext=(6.5, 1.15),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13', xy=(6, 1.5), xytext=(6.5, 1.6),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(6, 0.35), xytext=(6.5, 0.45),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("VGG and ResNet Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("./picture_data/VGG_Res_loss.png")
plt.show()


'''
2.前三个网络的acc
'''
plt.plot(np.arange(1, N+1), VGG11_acc, label="VGG11_acc(middle)")
plt.plot(np.arange(1, N+1), VGG13_acc, label="VGG13_acc(down)")
plt.plot(np.arange(1, N+1), Resnet18_acc, label="ResNet18_acc(up)")

plt.annotate('VGG11', xy=(7.5, 0.68), xytext=(8, 0.63),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13', xy=(7.5, 0.5), xytext=(8, 0.45),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(7.5, 0.9), xytext=(8, 0.85),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("VGG and ResNet Accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/VGG_Res_accuracy.png")
plt.show()


'''
3.前三个网络的val_loss
'''
plt.plot(np.arange(1, N+1), VGG11_val_loss, label="VGG11_val_loss")
plt.plot(np.arange(1, N+1), VGG13_val_loss, label="VGG13_val_loss")
plt.plot(np.arange(1, N+1), Resnet18_val_loss, label="ResNet18_val_loss")

plt.annotate('VGG11', xy=(1.2, 8.5), xytext=(2.2, 8.5),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13', xy=(2.5,5.5), xytext=(3.5,5.5),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(2, 0.6), xytext=(0.5, -0.3),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("VGG and ResNet Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("val_Loss")
plt.legend(loc="upper center")
plt.savefig("./picture_data/VGG_Res_val_Loss.png")
plt.show()


'''
4.前三个网络的val_acc
'''
plt.plot(np.arange(1, N+1), VGG11_val_acc, label="VGG11_val_acc(middle)")
plt.plot(np.arange(1, N+1), VGG13_val_acc, label="VGG13_val_acc(dowm)")
plt.plot(np.arange(1, N+1), Resnet18_val_acc, label="ResNet18_val_acc(up)")

plt.annotate('VGG11', xy=(5, 0.65), xytext=(6, 0.66),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13', xy=(4.5, 0.49), xytext=(5.5, 0.5),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(2.5, 0.9), xytext=(0.5, 0.95),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("VGG and ResNet Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("val_acc")
plt.legend(loc="lower right")
plt.savefig("./picture_data/VGG_Res_val_acc.png")
plt.show()


'''
5.RES和mobile的loss
'''
plt.plot(np.arange(1, N+1), Mobilenet_loss, label="MobileNetV1_loss")
plt.plot(np.arange(1, N+1), Resnet18_loss, label="ResNet18_loss")


plt.annotate('MobileNetV1', xy=(3.5, 0.95), xytext=(4, 1.05),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(3.5, 0.45), xytext=(1.5, 0.2),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("MobileNetV1 and ResNet Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("./picture_data/Mobile_Res_loss.png")
plt.show()


'''
6.RES和mobile的acc
'''
plt.plot(np.arange(1, N+1), Mobilenet_acc, label="MobileNetV1_acc")
plt.plot(np.arange(1, N+1), Resnet18_acc, label="ResNet18_acc")

plt.annotate('MobileNetV1', xy=(3.5, 0.68), xytext=(4, 0.63),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(3, 0.85), xytext=(0.5, 0.90),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("MobileNetV1 and ResNet Accuracy")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/Mobile_Res_accuracy.png")
plt.show()


'''
7.RES和mobile的val_loss
'''
plt.plot(np.arange(1, N+1), Mobilenet_val_loss, label="MobileNetV1_val_loss")
plt.plot(np.arange(1, N+1), Resnet18_val_loss, label="ResNet18_val_loss")

plt.annotate('MobileNetV1', xy=(1.3,1.77), xytext=(2.3, 1.85),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(2.5, 0.4), xytext=(0.5, 0.2),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("MobileNetV1 and ResNet Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("val_Loss")
plt.legend(loc="center right")
plt.savefig("./picture_data/Mobile_Res_val_Loss.png")
plt.show()


'''
8.RES和mobile的val_acc
'''
plt.plot(np.arange(1, N+1), Mobilenet_val_acc, label="MobileNetV1_val_acc")
plt.plot(np.arange(1, N+1), Resnet18_val_acc, label="ResNet18_val_acc")

plt.annotate('MobileNetV1',xy=(3, 0.68), xytext=(3.5, 0.63),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('ResNet18', xy=(3, 0.92), xytext=(0.5, 0.96),
             arrowprops=dict(arrowstyle="->",color="black"),
             )

plt.title("MobileNetV1 and ResNet Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("val_acc")
plt.legend(loc="lower right")
plt.savefig("./picture_data/Mobile_Res_val_acc.png")
plt.show()

'''
9.VGG11
'''
plt.plot(np.arange(1, N+1), VGG11_loss, label="VGG11_loss")
plt.plot(np.arange(1, N+1), VGG11_val_loss, label="VGG11_val_loss")

plt.annotate('VGG11_loss',xy=(7, 1.2), xytext=(8, 1.5),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG11_val_loss', xy=(2.8,5.0), xytext=(3.8, 5.3),
             arrowprops=dict(arrowstyle="->",color="black"),
             )


plt.title("VGG11 Test and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("./picture_data/VGG11_loss.png")
plt.show()


plt.plot(np.arange(1, N+1), VGG11_acc, label="VGG11_acc")
plt.plot(np.arange(1, N+1), VGG11_val_acc, label="VGG11_val_acc")
plt.annotate('VGG11_acc',xy=(5.4, 0.6), xytext=(6.4, 0.55),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG11_val_acc', xy=(6, 0.8), xytext=(3, 0.85),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("VGG11 Test and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/VGG11_acc.png")
plt.show()


'''
10.VGG13
'''
plt.plot(np.arange(1, N+1), VGG13_loss, label="VGG13_loss")
plt.plot(np.arange(1, N+1), VGG13_val_loss, label="VGG13_val_loss")

plt.annotate('VGG13_loss',xy=(4, 1.7), xytext=(5, 2.0),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13_val_loss', xy=(2.8,5.0), xytext=(3.8, 5.3),
             arrowprops=dict(arrowstyle="->",color="black"),
             )


plt.title("VGG13 Test and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper center")
plt.savefig("./picture_data/VGG13_loss.png")
plt.show()


plt.plot(np.arange(1, N+1), VGG13_acc, label="VGG13_acc")
plt.plot(np.arange(1, N+1), VGG13_val_acc, label="VGG13_val_acc")
plt.annotate('VGG13_acc',xy=(7.5, 0.5), xytext=(8, 0.45),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('VGG13_val_acc', xy=(4, 0.53), xytext=(1, 0.58),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("VGG13 Test and Validation Accuracy(epoch=15,batch_size=8)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/VGG13_acc.png")
plt.show()


'''
11.ResNet18
'''
plt.plot(np.arange(1, N+1), Resnet18_loss, label="Resnet18_loss")
plt.plot(np.arange(1, N+1), Resnet18_val_loss, label="Resnet18_val_loss")

plt.annotate('Resnet18_loss',xy=(2,1.1), xytext=(2.5,1.2),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('Resnet18_val_loss', xy=(4,0.2), xytext=(0.4, 0.05),
             arrowprops=dict(arrowstyle="->",color="black"),
             )


plt.title("Resnet18 Test and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("./picture_data/Res_loss.png")
plt.show()


plt.plot(np.arange(1, N+1), Resnet18_acc, label="Resnet18_acc")
plt.plot(np.arange(1, N+1), Resnet18_val_acc, label="Resnet18_val_acc")
plt.annotate('Resnet18_acc',xy=(2, 0.5), xytext=(2.5, 0.45),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('Resnet18_val_acc', xy=(4, 0.95), xytext=(1, 0.99),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("Resnet18 Test and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/Res_acc.png")
plt.show()

'''
12.MobileNet
'''

plt.plot(np.arange(1, N+1), Mobilenet_loss, label="MobilenetV1_loss")
plt.plot(np.arange(1, N+1), Mobilenet_val_loss, label="MobilenetV1_val_loss")

plt.annotate('MobilenetV1_loss',xy=(3,1.1), xytext=(3.6,1.2),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('MobilenetV1_val_loss', xy=(4,0.3), xytext=(0.4, 0.1),
             arrowprops=dict(arrowstyle="->",color="black"),
             )


plt.title("MobilenetV1 Test and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig("./picture_data/Mobile_loss.png")
plt.show()


plt.plot(np.arange(1, N+1), Mobilenet_acc, label="MobilenetV1_acc")
plt.plot(np.arange(1, N+1), Mobilenet_val_acc, label="MobilenetV1_val_acc")
plt.annotate('MobilenetV1_acc',xy=(2.5, 0.5), xytext=(3.5, 0.45),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.annotate('MobilenetV1_val_acc', xy=(4, 0.93), xytext=(1, 0.98),
             arrowprops=dict(arrowstyle="->",color="black"),
             )
plt.title("MobilenetV1 Test and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./picture_data/Mobile_acc.png")
plt.show()

