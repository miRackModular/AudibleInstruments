#include "AudibleInstruments.hpp"


Plugin *pluginInstance;

void init(rack::Plugin *p) {
	pluginInstance = p;

	p->addModel(modelBraids);
	p->addModel(modelPlaits);
	p->addModel(modelElements);
	p->addModel(modelTides);
	p->addModel(modelSheep);
	p->addModel(modelTides2);
	p->addModel(modelClouds);
	p->addModel(modelWarps);
	p->addModel(modelRings);
	p->addModel(modelLinks);
	p->addModel(modelKinks);
	p->addModel(modelShades);
	p->addModel(modelBranches);
	p->addModel(modelBlinds);
	p->addModel(modelVeils);
	p->addModel(modelFrames);
	p->addModel(modelPeaks);
	p->addModel(modelMarbles);
	p->addModel(modelStages);
}
