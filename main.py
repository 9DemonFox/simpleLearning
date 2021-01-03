from UI.Controler import Controler
from UI.Modeler import Modeler
from UI.Viewer import Viewer

if __name__ == '__main__':
    model = Modeler()
    view = Viewer()
    controler = Controler(model, view)
    controler.view.run()