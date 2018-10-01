TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    tools.cpp \
    networks.cpp \
    multilayer.cpp \
    recall.cpp

LIBS += -llapack -lblas -larmadillo

HEADERS += \
    tools.h \
    networks.h \
    multilayer.h \
    recall.h
