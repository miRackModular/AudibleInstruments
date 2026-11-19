#include "AudibleInstruments.hpp"
#include "beads/dsp/processor.h"
#include "beads/dsp/synth.h"
#include "beads/pot_controller.h"
#include "stmlib/dsp/dsp.h"
#include "stmlib/utils/gate_flags.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <vector>


static const int kBlockSize = 12;
static const int kSeedDelayCapacity = 1024;
static const float kSeedDelaySeconds = 0.0025f;
static const float kSeedHoldSeconds = 1.0f;
static const float kMediaHoldSeconds = 1.5f;
static const float kAssignHoldSeconds = 1.0f;
static const float kNoInputSeconds = 10.0f;
static const size_t kFrozenBytes = beads::kDelayBufferSizeBits / 8;
static const size_t kFrozenBank = 262144;
static const size_t kFrozenChunk = 4096;
static const char kFrozenMagic[4] = {'B', 'F', 'Z', '1'};


struct FrozenBuffer {
	int32_t writePtr = 0;
	int32_t media = 3;
	int32_t mono = 1;
	std::vector<uint8_t> bytes;
};


static void transferFrozenBytes(beads::Processor *processor, uint8_t *bytes, size_t offset, size_t count, bool intoProcessor) {
	uint8_t *raw = processor->delay_buffer_raw_bytes(offset);
	if (intoProcessor)
		std::memcpy(raw, bytes + offset, count);
	else
		std::memcpy(bytes + offset, raw, count);
}


static bool readFrozenFile(const std::filesystem::path &path, FrozenBuffer &buffer) {
	std::ifstream in(path, std::ios::binary);
	if (!in)
		return false;
	char magic[4];
	int32_t writePtr = 0;
	int32_t media = 0;
	int32_t mono = 0;
	int32_t size = 0;
	in.read(magic, 4);
	in.read(reinterpret_cast<char*>(&writePtr), sizeof(writePtr));
	in.read(reinterpret_cast<char*>(&media), sizeof(media));
	in.read(reinterpret_cast<char*>(&mono), sizeof(mono));
	in.read(reinterpret_cast<char*>(&size), sizeof(size));
	if (!in || std::memcmp(magic, kFrozenMagic, 4) != 0)
		return false;
	if (size != (int32_t) kFrozenBytes || media < 0 || media > 3 || (mono != 0 && mono != 1))
		return false;
	if (writePtr < 0 || writePtr > 1000000)
		return false;
	buffer.bytes.resize(kFrozenBytes);
	in.read(reinterpret_cast<char*>(buffer.bytes.data()), kFrozenBytes);
	if (!in)
		return false;
	buffer.writePtr = writePtr;
	buffer.media = media;
	buffer.mono = mono;
	return true;
}


static bool writeFrozenFile(const std::filesystem::path &path, const FrozenBuffer &buffer) {
	std::ofstream out(path, std::ios::binary);
	if (!out)
		return false;
	int32_t writePtr = buffer.writePtr;
	int32_t media = buffer.media;
	int32_t mono = buffer.mono;
	int32_t size = (int32_t) buffer.bytes.size();
	out.write(kFrozenMagic, 4);
	out.write(reinterpret_cast<const char*>(&writePtr), sizeof(writePtr));
	out.write(reinterpret_cast<const char*>(&media), sizeof(media));
	out.write(reinterpret_cast<const char*>(&mono), sizeof(mono));
	out.write(reinterpret_cast<const char*>(&size), sizeof(size));
	out.write(reinterpret_cast<const char*>(buffer.bytes.data()), buffer.bytes.size());
	return (bool) out;
}


static float attenuverterLaw(float v) {
	v -= 0.5f;
	return (2.0f / 0.47f) * v * std::max(std::fabs(v) - 0.03f, 0.0f);
}


// Knob at 0.5 is unison. The detents are a 2nd, 4th, 5th, octave and two octaves.
static float detuneLaw(float v) {
	float detune = 2.0f * v - 1.0f;
	float sign = detune < 0.0f ? -1.0f : 1.0f;
	detune = detune * sign * 6.0f;
	MAKE_INTEGRAL_FRACTIONAL(detune);

	const float intervals[8] = {
		0.0f, 1.0f, 3.0f, 4.0f, 7.0f, 12.0f, 24.0f, 24.0f
	};

	if (detune_integral < 0)
		detune_integral = 0;
	if (detune_integral > 6)
		detune_integral = 6;
	detune_fractional = std::max((detune_fractional - 0.33333333f) * 1.5f, 0.0f);

	float a = intervals[detune_integral];
	float b = intervals[detune_integral + 1];
	return stmlib::Crossfade(a, b, stmlib::SmoothStep(detune_fractional)) * sign;
}


static float mixWithAssign(float base, float cv, float amount) {
	float value = base * 1.005f + (cv - 0.002f) * amount;
	return clamp(value, 0.0f, 1.0f);
}


static void slopeToward(float &state, float target, float rise, float fall) {
	float error = target - state;
	state += (error > 0.0f ? rise : fall) * error;
}


// Float version of the firmware input leveler. A gain setting of 0 is automatic.
// Settings 1..255 follow its fourth-power curve from unity to 40x.
struct InputGain {
	float peak = 0.5f;
	float agcPeak = 0.5f;
	float stableAgcPeak = 0.5f;
	float applied = 1.0f;
	float manualSmoothed = 1.0f;
	float softAmount = 0.0f;
	int clipMode = 0;
	bool automatic = true;
	float oneHertz = 1.0f / 48000.0f;
	size_t time = 0;
	size_t ignoreSamples = 48000;
	size_t activeSamples = 48000 * 6;

	void init(float sampleRate, int mode, bool reset) {
		if (reset) {
			peak = agcPeak = stableAgcPeak = 0.5f;
			applied = 1.0f;
			softAmount = 0.0f;
		}
		clipMode = mode;
		ignoreSamples = (size_t) (sampleRate * 1.0f);
		bool keepLearned = !reset && time >= activeSamples && activeSamples > 0;
		activeSamples = (size_t) (sampleRate * 6.0f);
		time = keepLearned ? activeSamples : 0;
		oneHertz = 1.0f / sampleRate;
	}

	void restart() {
		time = ignoreSamples;
	}

	bool adjusting() const {
		return automatic && time > 0 && time < activeSamples;
	}

	float meter() const {
		float intensity = peak * applied;
		if (intensity > 1.0f)
			return 1.0f;
		return (192.0f / 255.0f) * intensity * intensity;
	}

	void process(bool active, uint8_t gainSetting, bool leftOpen, bool rightOpen, beads::FloatFrame *frames, int size) {
		if (size <= 0)
			return;
		if (leftOpen && rightOpen) {
			for (int i = 0; i < size; i++)
				frames[i].l = frames[i].r = 0.0f;
			peak = 0.0f;
			agcPeak = stableAgcPeak = 0.5f;
			applied = 1.0f;
			time = 0;
			return;
		}

		float blockHz = float(size) * oneHertz;
		if (time < ignoreSamples)
			agcPeak = 0.5f;
		else if (time < activeSamples) {
			slopeToward(agcPeak, peak, 10.0f * blockHz, 2.0f * blockHz);
			stableAgcPeak = agcPeak;
		}
		else {
			float release = (peak > stableAgcPeak ? 0.5f : 0.01f) * blockHz;
			slopeToward(agcPeak, peak, 5.0f * blockHz, release);
		}

		if (active) {
			if (time > ignoreSamples || peak > 0.001f)
				time += (size_t) size;
			if (time > activeSamples)
				time = activeSamples;
		}

		automatic = gainSetting == 0;
		float manual = float(gainSetting) / 255.0f;
		manual *= manual;
		manual *= manual;
		manual = 1.0f + 39.0f * manual;
		manualSmoothed += 0.01f * (manual - manualSmoothed);

		float learned = std::min(0.9f / std::min(agcPeak, 0.9f), 40.0f);
		float target = automatic ? learned : manualSmoothed;

		float softTarget = 0.0f;
		if (clipMode == 1)
			softTarget = 1.0f;
		else if (clipMode == 2) {
			softTarget = 2.0f * (manualSmoothed / learned - 0.75f);
			softTarget = clamp(softTarget, 0.0f, 1.0f);
		}

		float gainStart = applied;
		float softStart = softAmount;
		applied = target;
		softAmount = softTarget;

		for (int i = 0; i < size; i++) {
			float l = leftOpen ? frames[i].r : frames[i].l;
			float r = rightOpen ? frames[i].l : frames[i].r;
			if (active) {
				float samplePeak = std::max(std::fabs(l), std::fabs(r));
				slopeToward(peak, samplePeak, 0.1f, 2.0f * oneHertz);
			}
			float blend = (float(i) + 0.5f) / float(size);
			float gain = gainStart + (target - gainStart) * blend;
			l *= gain;
			r *= gain;
			if (!automatic) {
				if (clipMode == 0) {
					l = clamp(l, -1.0f, 1.0f);
					r = clamp(r, -1.0f, 1.0f);
				}
				else {
					float amount = softStart + (softTarget - softStart) * blend;
					const float bias = 0.05f;
					float softL = (1.0f - bias) * stmlib::SoftClip(l + bias) - bias;
					float softR = (1.0f - bias) * stmlib::SoftClip(r + bias) - bias;
					l = clamp(l, -1.0f, 1.0f);
					r = clamp(r, -1.0f, 1.0f);
					l = stmlib::Crossfade(l, softL, amount);
					r = stmlib::Crossfade(r, softR, amount);
				}
			}
			frames[i].l = l;
			frames[i].r = r;
		}
	}
};


struct Beads : Module {
	enum ParamIds {
		TIME_PARAM,
		SIZE_PARAM,
		SHAPE_PARAM,
		PITCH_PARAM,
		DENSITY_PARAM,
		TIME_ATTENUVERTER_PARAM,
		SIZE_ATTENUVERTER_PARAM,
		SHAPE_ATTENUVERTER_PARAM,
		PITCH_ATTENUVERTER_PARAM,
		FEEDBACK_PARAM,
		DRY_WET_PARAM,
		REVERB_PARAM,
		IN_GAIN_PARAM,
		SEED_PARAM,
		FREEZE_PARAM,
		MEDIA_PARAM,
		ASSIGN_PARAM,
		FEEDBACK_AMOUNT_PARAM,
		DRY_WET_AMOUNT_PARAM,
		REVERB_AMOUNT_PARAM,
		NUM_PARAMS
	};
	enum InputIds {
		TIME_INPUT,
		SIZE_INPUT,
		SHAPE_INPUT,
		PITCH_INPUT,
		DENSITY_INPUT,
		ASSIGN_INPUT,
		SEED_INPUT,
		FREEZE_INPUT,
		IN_L_INPUT,
		IN_R_INPUT,
		NUM_INPUTS
	};
	enum OutputIds {
		OUT_L_OUTPUT,
		OUT_R_OUTPUT,
		NUM_OUTPUTS
	};
	enum LightIds {
		FREEZE_LIGHT,
		SEED_LIGHT,
		MEDIA_RED_LIGHT,
		MEDIA_GREEN_LIGHT,
		MEDIA_BLUE_LIGHT,
		ASSIGN_1_LIGHT,
		ASSIGN_2_LIGHT,
		ASSIGN_3_LIGHT,
		VU_LIGHT,
		NUM_LIGHTS
	};

	dsp::SampleRateConverter<2> inputSrc;
	dsp::SampleRateConverter<2> outputSrc;
	dsp::DoubleRingBuffer<dsp::Frame<2>, 256> inputBuffer;
	dsp::DoubleRingBuffer<dsp::Frame<2>, 256> outputBuffer;

	uint8_t *delayBuffer1 = NULL;
	uint8_t *delayBuffer2 = NULL;
	uint16_t *reverbBuffer = NULL;
	beads::Processor *processor = NULL;
	beads::Synth *synth = NULL;

	int media = 3;
	int assign = 0;
	bool gateOutput = false;
	bool freeze = false;
	bool seedLocked = true;
	bool synthMode = true;
	double noInputSeconds = 0.0;
	bool gainManual = false;
	std::atomic<int> editView{0};

	beads::PotController mixPots[3];
	InputGain inputGain;
	float preparedGainRate = 0.0f;
	int preparedClip = -1;
	bool restartGain = false;

	bool buttonDown[4] = {};
	bool buttonWasDown[4] = {};
	uint64_t buttonHeld[4] = {};
	bool comboUsed = false;
	bool gainEdited = false;
	bool amountEdited[3] = {};
	bool freezeBeforePress = false;
	double uiPhase = 0.0;
	double seedPhase = 0.0;
	float gainSeen = 1.0f;
	float amountSeen[3] = {1.0f, 0.0f, 0.5f};

	std::mutex frozenMutex;
	std::shared_ptr<const FrozenBuffer> frozenSnapshot;
	std::unique_ptr<FrozenBuffer> frozenBuild;
	size_t frozenBuildOffset = 0;
	double frozenStableSeconds = 0.0;
	std::atomic<bool> snapshotReady{false};
	std::atomic<bool> snapshotDiscard{false};

	bool seedHostDelay[kSeedDelayCapacity] = {};
	int seedHostPos = 0;
	stmlib::GateFlags delayedSeedFlags = stmlib::GATE_FLAG_LOW;
	bool seedRisePending = false;
	bool seedFallPending = false;
	beads::SeedButtonState currentSeedState = beads::SEED_BUTTON_STATE_LOCKED;

	Beads() {
		config(NUM_PARAMS, NUM_INPUTS, NUM_OUTPUTS, NUM_LIGHTS);
		configParam(TIME_PARAM, 0.0f, 1.0f, 0.5f, "Time");
		configParam(SIZE_PARAM, 0.0f, 1.0f, 0.5f, "Size");
		configParam(SHAPE_PARAM, 0.0f, 1.0f, 0.5f, "Shape");
		configParam(PITCH_PARAM, 0.0f, 1.0f, 0.5f, "Pitch");
		configParam(DENSITY_PARAM, 0.0f, 1.0f, 0.5f, "Density");
		configParam(TIME_ATTENUVERTER_PARAM, 0.0f, 1.0f, 0.5f, "Time attenuverter");
		configParam(SIZE_ATTENUVERTER_PARAM, 0.0f, 1.0f, 0.5f, "Size attenuverter");
		configParam(SHAPE_ATTENUVERTER_PARAM, 0.0f, 1.0f, 0.5f, "Shape attenuverter");
		configParam(PITCH_ATTENUVERTER_PARAM, 0.0f, 1.0f, 0.5f, "Pitch attenuverter");
		configParam(FEEDBACK_PARAM, 0.0f, 1.0f, 0.0f, "Feedback");
		configParam(DRY_WET_PARAM, 0.0f, 1.0f, 0.5f, "Dry/wet");
		configParam(REVERB_PARAM, 0.0f, 1.0f, 0.0f, "Reverb");
		configParam(IN_GAIN_PARAM, 0.0f, 1.0f, 1.0f, "Manual input gain");
		configParam(SEED_PARAM, 0.0f, 1.0f, 0.0f, "Seed");
		configParam(FREEZE_PARAM, 0.0f, 1.0f, 0.0f, "Freeze");
		configParam(MEDIA_PARAM, 0.0f, 1.0f, 0.0f, "Audio quality");
		configParam(ASSIGN_PARAM, 0.0f, 1.0f, 0.0f, "Assign");
		configParam(FEEDBACK_AMOUNT_PARAM, 0.0f, 1.0f, 1.0f, "Feedback CV amount");
		configParam(DRY_WET_AMOUNT_PARAM, 0.0f, 1.0f, 0.0f, "Dry/wet CV amount");
		configParam(REVERB_AMOUNT_PARAM, 0.0f, 1.0f, 0.5f, "Reverb CV amount");

		delayBuffer1 = new uint8_t[512 * 1024]();
		delayBuffer2 = new uint8_t[256 * 1024]();
		reverbBuffer = new uint16_t[32768]();
		processor = new beads::Processor();
		synth = new beads::Synth();
		std::memset(static_cast<void*>(processor), 0, sizeof(*processor));
		std::memset(static_cast<void*>(synth), 0, sizeof(*synth));

		uint8_t *buffers[3];
		buffers[0] = delayBuffer1;
		buffers[1] = delayBuffer1 + 256 * 1024;
		buffers[2] = delayBuffer2;
		processor->Init(buffers, reverbBuffer);
		synth->Init(reverbBuffer);
		onReset();
	}

	~Beads() {
		delete processor;
		delete synth;
		delete[] delayBuffer1;
		delete[] delayBuffer2;
		delete[] reverbBuffer;
	}

	bool frozenNow() {
		return freeze || inputs[FREEZE_INPUT].getVoltage() >= 1.0f;
	}

	void prepareGain(float sampleRate) {
		int clip = 0;
		if (media == 0)
			clip = 1;
		else if (media == 1)
			clip = 2;
		if (sampleRate == preparedGainRate && clip == preparedClip)
			return;
		bool first = preparedGainRate == 0.0f;
		inputGain.init(sampleRate, clip, first);
		preparedGainRate = sampleRate;
		preparedClip = clip;
	}

	void realignMixPots() {
		const int ids[3] = {FEEDBACK_PARAM, DRY_WET_PARAM, REVERB_PARAM};
		for (int i = 0; i < 3; i++) {
			mixPots[i].Init(1.0f, 0.0f);
			mixPots[i].ProcessControlRate(params[ids[i]].getValue());
			mixPots[i].ProcessUIRate();
		}
	}

	void unlockMixPot(int index) {
		const int ids[3] = {FEEDBACK_PARAM, DRY_WET_PARAM, REVERB_PARAM};
		if (mixPots[index].editing_hidden_parameter()) {
			float audible = mixPots[index].main_parameter();
			params[ids[index]].setValue(audible);
			mixPots[index].Init(1.0f, 0.0f);
			mixPots[index].ProcessControlRate(audible);
			mixPots[index].ProcessUIRate();
		}
		else {
			mixPots[index].Unlock();
		}
	}

	void applyEngineSettings() {
		if (frozenNow())
			return;
		bool left = inputs[IN_L_INPUT].isConnected();
		bool right = inputs[IN_R_INPUT].isConnected();
		// Both jacks empty is the oscillator countdown. Leave the recording
		// buffer alone until an input returns.
		if (!left && !right)
			return;
		bool mono = !(left && right);
		beads::Processor::Media engineMedia = static_cast<beads::Processor::Media>(media);
		if (processor->media() != engineMedia || processor->mono() != mono)
			processor->Configure(engineMedia, mono);
	}

	void readSeed(float sampleRate) {
		bool high = inputs[SEED_INPUT].getVoltage() >= 1.0f;
		int delaySamples = (int) std::lround(kSeedDelaySeconds * sampleRate);
		if (delaySamples < 1)
			delaySamples = 1;
		if (delaySamples >= kSeedDelayCapacity)
			delaySamples = kSeedDelayCapacity - 1;

		seedHostDelay[seedHostPos] = high;
		int readPos = (seedHostPos - delaySamples + kSeedDelayCapacity) & (kSeedDelayCapacity - 1);
		bool delayedHigh = seedHostDelay[readPos];
		seedHostPos = (seedHostPos + 1) & (kSeedDelayCapacity - 1);

		delayedSeedFlags = stmlib::ExtractGateFlags(delayedSeedFlags, delayedHigh);
		if (delayedSeedFlags & stmlib::GATE_FLAG_RISING)
			seedRisePending = true;
		if (delayedSeedFlags & stmlib::GATE_FLAG_FALLING)
			seedFallPending = true;
	}

	void updateButtons(const ProcessArgs &args) {
		const int ids[4] = {MEDIA_PARAM, FREEZE_PARAM, SEED_PARAM, ASSIGN_PARAM};
		bool pressed[4];
		for (int i = 0; i < 4; i++)
			pressed[i] = params[ids[i]].getValue() >= 1.0f;

		for (int i = 0; i < 4; i++) {
			if (pressed[i] && !buttonWasDown[i]) {
				buttonDown[i] = true;
				buttonHeld[i] = 0;
				if (i == 1)
					freezeBeforePress = freeze;
				if (!comboUsed && ((i == 1 && pressed[2]) || (i == 2 && pressed[1]))) {
					seedLocked = true;
					comboUsed = true;
					freeze = freezeBeforePress;
				}
				else if (!comboUsed && ((i == 1 && pressed[3]) || (i == 3 && pressed[1]))) {
					comboUsed = true;
					freeze = freezeBeforePress;
				}
				else if (!comboUsed && ((i == 2 && pressed[3]) || (i == 3 && pressed[2]))) {
					gateOutput = !gateOutput;
					comboUsed = true;
				}
				if (i == 2 && !comboUsed && seedLocked)
					seedLocked = false;
				if (i == 1 && !comboUsed)
					freeze = !freeze;
				if (i == 0) {
					mixPots[0].Lock();
					gainEdited = false;
				}
				if (i == 3) {
					for (int k = 0; k < 3; k++) {
						mixPots[k].Lock();
						amountEdited[k] = false;
					}
				}
			}
			if (buttonDown[i] && pressed[i])
				buttonHeld[i]++;
			if (!pressed[i] && buttonWasDown[i]) {
				buttonDown[i] = false;
				if (!comboUsed) {
					if (i == 0 && !gainEdited) {
						if (buttonHeld[i] >= (uint64_t) (args.sampleRate * kMediaHoldSeconds)) {
							gainManual = false;
							restartGain = true;
						}
						else {
							media = (media + 3) & 3;
						}
					}
					else if (i == 2) {
						seedLocked = buttonHeld[i] >= (uint64_t) (args.sampleRate * kSeedHoldSeconds);
					}
					else if (i == 3) {
						bool edited = amountEdited[0] || amountEdited[1] || amountEdited[2];
						if (!edited && buttonHeld[i] < (uint64_t) (args.sampleRate * kAssignHoldSeconds))
							assign = assign == 3 ? 0 : (assign + 1) % 3;
					}
				}
				if (i == 0)
					unlockMixPot(0);
				if (i == 3) {
					for (int k = 0; k < 3; k++)
						unlockMixPot(k);
				}
				buttonHeld[i] = 0;
			}
			if (i == 3 && buttonDown[i] && !comboUsed && buttonHeld[i] == (uint64_t) (args.sampleRate * kAssignHoldSeconds)) {
				int view = editView.load();
				editView.store(view == 2 ? 0 : 2);
			}
			buttonWasDown[i] = pressed[i];
		}
		if (comboUsed && !pressed[1] && !pressed[2] && !pressed[3])
			comboUsed = false;

		if (seedLocked)
			currentSeedState = beads::SEED_BUTTON_STATE_LOCKED;
		else if (pressed[2])
			currentSeedState = beads::SEED_BUTTON_STATE_ON;
		else
			currentSeedState = beads::SEED_BUTTON_STATE_OFF;
	}

	void updateHiddenKnobs() {
		const int amountIds[3] = {FEEDBACK_AMOUNT_PARAM, DRY_WET_AMOUNT_PARAM, REVERB_AMOUNT_PARAM};
		if (buttonDown[0] && mixPots[0].editing_hidden_parameter()) {
			gainEdited = true;
			gainManual = true;
			float hidden = clamp(mixPots[0].hidden_parameter(), 0.0f, 1.0f);
			params[IN_GAIN_PARAM].setValue(hidden);
			gainSeen = hidden;
		}
		if (buttonDown[3]) {
			for (int i = 0; i < 3; i++) {
				if (mixPots[i].editing_hidden_parameter()) {
					amountEdited[i] = true;
					float hidden = clamp(mixPots[i].hidden_parameter(), 0.0f, 1.0f);
					params[amountIds[i]].setValue(hidden);
					amountSeen[i] = hidden;
					assign = 3;
				}
			}
		}

		float gainKnob = params[IN_GAIN_PARAM].getValue();
		if (editView.load() == 1 && std::fabs(gainKnob - gainSeen) > 0.0001f)
			gainManual = true;
		gainSeen = gainKnob;

		if (editView.load() == 2) {
			for (int i = 0; i < 3; i++) {
				float amount = params[amountIds[i]].getValue();
				if (std::fabs(amount - amountSeen[i]) > 0.0001f)
					assign = 3;
				amountSeen[i] = amount;
			}
		}
	}

	void fillParameters(beads::Parameters *parameters, beads::ModulatedParameter *density, beads::Processor::MixParameters *mix) {
		parameters->time.base = params[TIME_PARAM].getValue();
		parameters->time.attenuverter = attenuverterLaw(params[TIME_ATTENUVERTER_PARAM].getValue());
		parameters->time.has_cv = inputs[TIME_INPUT].isConnected();
		parameters->time.cv = parameters->time.has_cv ? clamp(inputs[TIME_INPUT].getVoltage() / 5.0f, -1.0f, 1.0f) * 1.6f : 0.0f;

		parameters->size.base = params[SIZE_PARAM].getValue();
		parameters->size.attenuverter = attenuverterLaw(params[SIZE_ATTENUVERTER_PARAM].getValue());
		parameters->size.has_cv = inputs[SIZE_INPUT].isConnected();
		parameters->size.cv = parameters->size.has_cv ? clamp(inputs[SIZE_INPUT].getVoltage() / 5.0f, -1.0f, 1.0f) * 1.6f : 0.0f;

		parameters->shape.base = params[SHAPE_PARAM].getValue();
		parameters->shape.attenuverter = attenuverterLaw(params[SHAPE_ATTENUVERTER_PARAM].getValue());
		parameters->shape.has_cv = inputs[SHAPE_INPUT].isConnected();
		parameters->shape.cv = parameters->shape.has_cv ? clamp(inputs[SHAPE_INPUT].getVoltage() / 5.0f, -1.0f, 1.0f) * 1.6f : 0.0f;

		parameters->pitch.base = detuneLaw(params[PITCH_PARAM].getValue());
		parameters->pitch.attenuverter = attenuverterLaw(params[PITCH_ATTENUVERTER_PARAM].getValue());
		parameters->pitch.has_cv = inputs[PITCH_INPUT].isConnected();
		parameters->pitch.cv = parameters->pitch.has_cv ? clamp(inputs[PITCH_INPUT].getVoltage() * 12.0f, -60.0f, 60.0f) : 0.0f;
		parameters->monophonic_mode = false;

		density->base = params[DENSITY_PARAM].getValue();
		density->cv = inputs[DENSITY_INPUT].isConnected() ? clamp(inputs[DENSITY_INPUT].getVoltage() / 5.0f, -1.0f, 1.0f) : 0.0f;

		float assignCv = inputs[ASSIGN_INPUT].isConnected() ? clamp(inputs[ASSIGN_INPUT].getVoltage() / 5.0f, -1.0f, 1.0f) : 0.0f;
		const int amountIds[3] = {FEEDBACK_AMOUNT_PARAM, DRY_WET_AMOUNT_PARAM, REVERB_AMOUNT_PARAM};
		float amount[3];
		for (int i = 0; i < 3; i++)
			amount[i] = assign == 3 ? params[amountIds[i]].getValue() : (assign == i ? 1.0f : 0.0f);
		mix->feedback = mixWithAssign(mixPots[0].main_parameter(), assignCv, amount[0]);
		mix->dry_wet = mixWithAssign(mixPots[1].main_parameter(), assignCv, amount[1]);
		mix->reverb_amount = mixWithAssign(mixPots[2].main_parameter(), assignCv, amount[2]);
		mix->output_gate = gateOutput;
	}

	std::filesystem::path frozenBufferPath(bool create) {
		return std::filesystem::u8path(getPatchStorageDirectory(create)) / "frozen-buffer.bin";
	}

	void requestDiscardFrozenSnapshot() {
		snapshotDiscard.store(true);
		std::lock_guard<std::mutex> lock(frozenMutex);
		frozenSnapshot.reset();
		snapshotReady.store(false);
	}

	void installFrozenBuffer(const FrozenBuffer &buffer) {
		processor->Configure(static_cast<beads::Processor::Media>(buffer.media), buffer.mono != 0);
		size_t offset = 0;
		while (offset < kFrozenBytes) {
			size_t bankRemain = kFrozenBank - (offset % kFrozenBank);
			size_t count = std::min(bankRemain, kFrozenBytes - offset);
			transferFrozenBytes(processor, const_cast<uint8_t*>(buffer.bytes.data()), offset, count, true);
			offset += count;
		}
		*processor->mutable_delay_write_ptr() = buffer.writePtr;
		media = buffer.media;
		freeze = true;
		synthMode = false;
		frozenBuild.reset();
		frozenBuildOffset = 0;
		frozenStableSeconds = 0.0;
		{
			std::lock_guard<std::mutex> lock(frozenMutex);
			frozenSnapshot = std::make_shared<FrozenBuffer>(buffer);
			snapshotReady.store(true);
		}
		snapshotDiscard.store(false);
	}

	void copyFrozenChunk() {
		if (!frozenBuild || snapshotDiscard.load())
			return;
		size_t count = std::min(kFrozenChunk, kFrozenBytes - frozenBuildOffset);
		size_t bankRemain = kFrozenBank - (frozenBuildOffset % kFrozenBank);
		count = std::min(count, bankRemain);
		transferFrozenBytes(processor, frozenBuild->bytes.data(), frozenBuildOffset, count, false);
		frozenBuildOffset += count;
		if (frozenBuildOffset < kFrozenBytes)
			return;
		if (snapshotDiscard.load()) {
			frozenBuild.reset();
			frozenBuildOffset = 0;
			return;
		}
		std::lock_guard<std::mutex> lock(frozenMutex);
		if (snapshotDiscard.load()) {
			frozenBuild.reset();
			frozenBuildOffset = 0;
			return;
		}
		frozenSnapshot = std::shared_ptr<const FrozenBuffer>(std::move(frozenBuild));
		snapshotReady.store(true);
	}

	void maintainFrozenSnapshot(float sampleTime) {
		if (snapshotDiscard.exchange(false)) {
			frozenBuild.reset();
			frozenBuildOffset = 0;
			frozenStableSeconds = 0.0;
			std::lock_guard<std::mutex> lock(frozenMutex);
			frozenSnapshot.reset();
			snapshotReady.store(false);
		}
		if (synthMode || !frozenNow()) {
			frozenBuild.reset();
			frozenBuildOffset = 0;
			frozenStableSeconds = 0.0;
			if (snapshotReady.exchange(false)) {
				std::lock_guard<std::mutex> lock(frozenMutex);
				frozenSnapshot.reset();
			}
			return;
		}
		if (snapshotReady.load())
			return;
		frozenStableSeconds += sampleTime;
		if (frozenStableSeconds < 0.02)
			return;
		if (frozenBuild)
			return;
		frozenBuild = std::unique_ptr<FrozenBuffer>(new FrozenBuffer());
		frozenBuild->bytes.resize(kFrozenBytes);
		frozenBuild->writePtr = *processor->mutable_delay_write_ptr();
		frozenBuild->media = (int) processor->media();
		frozenBuild->mono = processor->mono() ? 1 : 0;
		frozenBuildOffset = 0;
	}

	void renderBlock(const ProcessArgs &args) {
		copyFrozenChunk();
		float engineRate = synthMode ? 48000.0f : processor->sample_rate();
		prepareGain(engineRate);
		if (restartGain) {
			inputGain.restart();
			restartGain = false;
		}

		dsp::Frame<2> inputFrames[kBlockSize];
		int inLen = inputBuffer.size();
		int outLen = kBlockSize;
		inputSrc.setRates(args.sampleRate, engineRate);
		inputSrc.process(inputBuffer.startData(), &inLen, inputFrames, &outLen);
		inputBuffer.startIncr(inLen);
		if (outLen <= 0)
			return;

		beads::FloatFrame input[kBlockSize];
		beads::FloatFrame output[kBlockSize];
		bool freezeBlock[kBlockSize];
		stmlib::GateFlags seedBlock[kBlockSize];
		bool frozen = frozenNow();
		bool leftOpen = !inputs[IN_L_INPUT].isConnected();
		bool rightOpen = !inputs[IN_R_INPUT].isConnected();
		for (int i = 0; i < outLen; i++) {
			input[i].l = inputFrames[i].samples[0];
			input[i].r = inputFrames[i].samples[1];
		}
		uint8_t gainSetting = 0;
		if (gainManual) {
			int setting = (int) std::lround(params[IN_GAIN_PARAM].getValue() * 255.0f);
			gainSetting = (uint8_t) clamp(setting, 1, 255);
		}
		inputGain.process(true, gainSetting, leftOpen, rightOpen, input, outLen);

		for (int i = 0; i < outLen; i++) {
			freezeBlock[i] = frozen;
			if (seedRisePending) {
				seedBlock[i] = stmlib::GATE_FLAG_RISING | stmlib::GATE_FLAG_HIGH;
				seedRisePending = false;
			}
			else if (seedFallPending) {
				seedBlock[i] = stmlib::GATE_FLAG_FALLING;
				seedFallPending = false;
			}
			else {
				seedBlock[i] = (delayedSeedFlags & stmlib::GATE_FLAG_HIGH) ? stmlib::GATE_FLAG_HIGH : stmlib::GATE_FLAG_LOW;
			}
		}

		beads::Parameters parameters;
		beads::ModulatedParameter density;
		beads::Processor::MixParameters mix;
		fillParameters(&parameters, &density, &mix);

		// A non-null seed pointer means the jack is patched. While latched, that
		// selects the clock divider instead of the free-running grain clock.
		const stmlib::GateFlags *seedInput = inputs[SEED_INPUT].isConnected() ? seedBlock : NULL;

		if (synthMode) {
			synth->Render(
				currentSeedState,
				seedInput,
				freezeBlock,
				static_cast<beads::Synth::Resolution>(media),
				density,
				parameters,
				mix,
				output,
				outLen);
		}
		else {
			processor->Process(
				currentSeedState,
				seedInput,
				freezeBlock,
				density,
				parameters,
				mix,
				input,
				output,
				outLen);
			for (int i = 0; i < outLen; i++) {
				if (std::fabs(output[i].l) > 0.01f || std::fabs(output[i].r) > 0.01f) {
					noInputSeconds = 0.0;
					break;
				}
			}
		}

		dsp::Frame<2> outputFrames[kBlockSize];
		for (int i = 0; i < outLen; i++) {
			outputFrames[i].samples[0] = output[i].l;
			outputFrames[i].samples[1] = output[i].r;
		}
		outputSrc.setRates(engineRate, args.sampleRate);
		int outputInLen = outLen;
		int outputOutLen = outputBuffer.capacity();
		outputSrc.process(outputFrames, &outputInLen, outputBuffer.endData(), &outputOutLen);
		outputBuffer.endIncr(outputOutLen);
	}

	void updateLights(const ProcessArgs &args) {
		seedPhase += args.sampleTime / 2.048;
		if (seedPhase >= 1.0)
			seedPhase -= 1.0;
		float triangle = float(1.0 - std::fabs(2.0 * seedPhase - 1.0));

		bool seedButton = params[SEED_PARAM].getValue() >= 1.0f;
		bool seedGate = inputs[SEED_INPUT].getVoltage() >= 1.0f;
		float seedBrightness = 0.0f;
		if (seedButton || seedGate)
			seedBrightness = 1.0f;
		else if (seedLocked)
			seedBrightness = 0.55f + 0.45f * triangle;
		lights[SEED_LIGHT].setBrightness(seedBrightness);
		lights[FREEZE_LIGHT].setBrightness(frozenNow() ? 1.0f : 0.0f);

		const int amountIds[3] = {FEEDBACK_AMOUNT_PARAM, DRY_WET_AMOUNT_PARAM, REVERB_AMOUNT_PARAM};
		for (int i = 0; i < 3; i++) {
			float level = 0.0f;
			if (assign == i)
				level = 0.8f;
			else if (assign == 3)
				level = params[amountIds[i]].getValue();
			lights[ASSIGN_1_LIGHT + i].setBrightness(level);
		}

		bool pendingMedia = (int) processor->media() != media;
		float mediaLevel = pendingMedia && ((int) (seedPhase * 8.192) & 1) ? 0.25f : 0.85f;
		const float red[4] = {128.0f / 255.0f, 1.0f, 0.0f, 192.0f / 255.0f};
		const float green[4] = {0.0f, 144.0f / 255.0f, 1.0f, 192.0f / 255.0f};
		const float blue[4] = {48.0f / 255.0f, 0.0f, 192.0f / 255.0f, 208.0f / 255.0f};
		lights[MEDIA_RED_LIGHT].setBrightness(red[media] * mediaLevel);
		lights[MEDIA_GREEN_LIGHT].setBrightness(green[media] * mediaLevel);
		lights[MEDIA_BLUE_LIGHT].setBrightness(blue[media] * mediaLevel);

		bool inputsPatched = inputs[IN_L_INPUT].isConnected() || inputs[IN_R_INPUT].isConnected();
		if (inputGain.adjusting() && inputsPatched)
			lights[VU_LIGHT].setBrightness(((int) (seedPhase * 16.0) & 1) ? 0.5f : 0.0f);
		else
			lights[VU_LIGHT].setBrightness(inputGain.meter());
	}

	void process(const ProcessArgs &args) override {
		const int mixIds[3] = {FEEDBACK_PARAM, DRY_WET_PARAM, REVERB_PARAM};
		for (int i = 0; i < 3; i++)
			mixPots[i].ProcessControlRate(params[mixIds[i]].getValue());
		uiPhase += 1000.0 / std::max(args.sampleRate, 1.0f);
		if (uiPhase >= 1.0) {
			uiPhase -= 1.0;
			for (int i = 0; i < 3; i++)
				mixPots[i].ProcessUIRate();
		}

		updateHiddenKnobs();
		updateButtons(args);
		applyEngineSettings();
		readSeed(args.sampleRate);

		bool anyInput = inputs[IN_L_INPUT].isConnected() || inputs[IN_R_INPUT].isConnected();
		if (anyInput || frozenNow())
			noInputSeconds = 0.0;
		else
			noInputSeconds += args.sampleTime;
		if (!frozenNow() && anyInput)
			synthMode = false;

		if (!inputBuffer.full()) {
			dsp::Frame<2> inputFrame = {};
			inputFrame.samples[0] = inputs[IN_L_INPUT].isConnected() ? inputs[IN_L_INPUT].getVoltage() / 5.0f : 0.0f;
			inputFrame.samples[1] = inputs[IN_R_INPUT].isConnected() ? inputs[IN_R_INPUT].getVoltage() / 5.0f : 0.0f;
			inputBuffer.push(inputFrame);
		}

		if (outputBuffer.empty())
			renderBlock(args);

		if (!frozenNow() && !anyInput && noInputSeconds >= kNoInputSeconds)
			synthMode = true;

		maintainFrozenSnapshot(args.sampleTime);

		if (!outputBuffer.empty()) {
			dsp::Frame<2> outputFrame = outputBuffer.shift();
			// Inverting output stage. A gate stored as -0.8 becomes a positive gate.
			float left = -5.0f * outputFrame.samples[0];
			float right = -5.0f * outputFrame.samples[1];
			if (!outputs[OUT_R_OUTPUT].isConnected() && !gateOutput)
				left = (left + right) * 0.5f;
			outputs[OUT_L_OUTPUT].setVoltage(left);
			outputs[OUT_R_OUTPUT].setVoltage(right);
		}

		updateLights(args);
	}

	void onAdd(const AddEvent &e) override {
		FrozenBuffer buffer;
		if (!readFrozenFile(frozenBufferPath(false), buffer))
			return;
		installFrozenBuffer(buffer);
	}

	void onSave(const SaveEvent &e) override {
		std::shared_ptr<const FrozenBuffer> buffer;
		{
			std::lock_guard<std::mutex> lock(frozenMutex);
			if (snapshotReady.load())
				buffer = frozenSnapshot;
		}
		std::error_code ec;
		std::filesystem::path path = frozenBufferPath(true);
		if (!buffer) {
			std::filesystem::remove(path, ec);
			return;
		}
		std::filesystem::path temporary = path;
		temporary += ".tmp";
		if (!writeFrozenFile(temporary, *buffer)) {
			std::filesystem::remove(temporary, ec);
			return;
		}
		std::filesystem::rename(temporary, path, ec);
		if (ec) {
			std::filesystem::remove(path, ec);
			std::filesystem::rename(temporary, path, ec);
		}
	}

	void onReset() override {
		requestDiscardFrozenSnapshot();
		media = 3;
		assign = 0;
		gateOutput = false;
		freeze = false;
		seedLocked = true;
		synthMode = true;
		noInputSeconds = 0.0;
		gainManual = false;
		editView.store(0);
		restartGain = false;
		comboUsed = false;
		gainEdited = false;
		preparedGainRate = 0.0f;
		preparedClip = -1;
		seedHostPos = 0;
		delayedSeedFlags = stmlib::GATE_FLAG_LOW;
		seedRisePending = false;
		seedFallPending = false;
		currentSeedState = beads::SEED_BUTTON_STATE_LOCKED;
		std::memset(seedHostDelay, 0, sizeof(seedHostDelay));
		std::memset(buttonDown, 0, sizeof(buttonDown));
		std::memset(buttonWasDown, 0, sizeof(buttonWasDown));
		std::memset(buttonHeld, 0, sizeof(buttonHeld));
		std::memset(amountEdited, 0, sizeof(amountEdited));
		params[IN_GAIN_PARAM].setValue(1.0f);
		params[FEEDBACK_AMOUNT_PARAM].setValue(1.0f);
		params[DRY_WET_AMOUNT_PARAM].setValue(0.0f);
		params[REVERB_AMOUNT_PARAM].setValue(0.5f);
		gainSeen = 1.0f;
		amountSeen[0] = 1.0f;
		amountSeen[1] = 0.0f;
		amountSeen[2] = 0.5f;
		realignMixPots();
		if (processor)
			processor->Configure(beads::Processor::MEDIA_DIGITAL, true);
	}

	json_t *dataToJson() override {
		json_t *rootJ = json_object();
		json_object_set_new(rootJ, "media", json_integer(media));
		json_object_set_new(rootJ, "assign", json_integer(assign));
		json_object_set_new(rootJ, "gateOutput", json_boolean(gateOutput));
		json_object_set_new(rootJ, "freeze", json_boolean(freeze));
		json_object_set_new(rootJ, "seedLock", json_boolean(seedLocked));
		json_object_set_new(rootJ, "gainManual", json_boolean(gainManual));
		return rootJ;
	}

	void dataFromJson(json_t *rootJ) override {
		json_t *mediaJ = json_object_get(rootJ, "media");
		if (mediaJ)
			media = clamp((int) json_integer_value(mediaJ), 0, 3);

		json_t *assignJ = json_object_get(rootJ, "assign");
		if (assignJ)
			assign = clamp((int) json_integer_value(assignJ), 0, 3);

		json_t *gateJ = json_object_get(rootJ, "gateOutput");
		if (gateJ)
			gateOutput = json_boolean_value(gateJ);

		json_t *freezeJ = json_object_get(rootJ, "freeze");
		if (freezeJ)
			freeze = json_boolean_value(freezeJ);

		json_t *seedJ = json_object_get(rootJ, "seedLock");
		if (seedJ)
			seedLocked = json_boolean_value(seedJ);

		json_t *gainJ = json_object_get(rootJ, "gainManual");
		if (gainJ)
			gainManual = json_boolean_value(gainJ);

		gainSeen = params[IN_GAIN_PARAM].getValue();
		amountSeen[0] = params[FEEDBACK_AMOUNT_PARAM].getValue();
		amountSeen[1] = params[DRY_WET_AMOUNT_PARAM].getValue();
		amountSeen[2] = params[REVERB_AMOUNT_PARAM].getValue();
		realignMixPots();
	}
};


struct BeadsQualityItem : MenuItem {
	Beads *module;
	int media;
	void onAction(const event::Action &e) override {
		module->media = media;
	}
	void step() override {
		rightText = (module->media == media) ? "✔" : "";
		MenuItem::step();
	}
};


struct BeadsToggleItem : MenuItem {
	Beads *module;
	int which;
	void onAction(const event::Action &e) override {
		if (which == 0)
			module->gateOutput = !module->gateOutput;
		else
			module->seedLocked = !module->seedLocked;
	}
	void step() override {
		bool on = which == 0 ? module->gateOutput : module->seedLocked;
		rightText = on ? "✔" : "";
		MenuItem::step();
	}
};


struct BeadsViewItem : MenuItem {
	Beads *module;
	int view;
	void onAction(const event::Action &e) override {
		module->editView.store(view);
	}
	void step() override {
		rightText = (module->editView.load() == view) ? "✔" : "";
		MenuItem::step();
	}
};


static size_t gainMenuIndex(Beads *module) {
	if (!module->gainManual)
		return 0;
	float value = clamp(module->params[Beads::IN_GAIN_PARAM].getValue(), 0.0f, 1.0f);
	return (size_t) std::lround(value * 4.0f) + 1;
}


static void setGainMenuIndex(Beads *module, size_t index) {
	if (index == 0) {
		module->gainManual = false;
		return;
	}
	float value = (index - 1) / 4.0f;
	module->params[Beads::IN_GAIN_PARAM].setValue(value);
	module->gainManual = true;
	module->gainSeen = value;
}


struct FreezeSeedLight : YellowLight {
	FreezeSeedLight() {
		box.size = Vec(28 - 6, 28 - 6);
		bgColor = color::BLACK_TRANSPARENT;
	}
};


struct BeadsWidget : ModuleWidget {
	ParamWidget *feedbackKnob;
	ParamWidget *dryWetKnob;
	ParamWidget *reverbKnob;
	ParamWidget *gainKnob;
	ParamWidget *amountKnobs[3];
	int currentView = -1;

	BeadsWidget(Beads *module) {
		setModule(module);
		setPanel(APP->window->loadSvg(asset::plugin(pluginInstance, "res/Beads.svg")));

		addChild(createWidget<ScrewSilver>(Vec(15, 0)));
		addChild(createWidget<ScrewSilver>(Vec(180, 0)));
		addChild(createWidget<ScrewSilver>(Vec(15, 365)));
		addChild(createWidget<ScrewSilver>(Vec(180, 365)));

		addParam(createParamCentered<TL1105>(Vec(104.67, 133.22), module, Beads::MEDIA_PARAM));
		addParam(createParamCentered<TL1105>(Vec(20.04, 227.11), module, Beads::ASSIGN_PARAM));
		addParam(createParamCentered<CKD6>(Vec(183.57, 52.49), module, Beads::SEED_PARAM));
		addParam(createParamCentered<CKD6>(Vec(25.81, 52.49), module, Beads::FREEZE_PARAM));

		addChild(createLightCentered<MediumLight<RedGreenBlueLight>>(Vec(85.84, 133.22), module, Beads::MEDIA_RED_LIGHT));
		addChild(createLightCentered<MediumLight<YellowLight>>(Vec(20.035, 206.415), module, Beads::ASSIGN_1_LIGHT));
		addChild(createLightCentered<MediumLight<YellowLight>>(Vec(189.335, 206.415), module, Beads::ASSIGN_2_LIGHT));
		addChild(createLightCentered<MediumLight<YellowLight>>(Vec(189.335, 254.315), module, Beads::ASSIGN_3_LIGHT));
		addChild(createLightCentered<MediumLight<GreenLight>>(Vec(37.045, 309.765), module, Beads::VU_LIGHT));
		addChild(createLightCentered<FreezeSeedLight>(Vec(183.57, 52.49), module, Beads::SEED_LIGHT));
		addChild(createLightCentered<FreezeSeedLight>(Vec(25.81, 52.49), module, Beads::FREEZE_LIGHT));

		addParam(createParamCentered<Rogan3PSWhite>(Vec(104.67, 71.27), module, Beads::DENSITY_PARAM));
		addParam(createParamCentered<Rogan3PSWhite>(Vec(34.23, 113.55), module, Beads::TIME_PARAM));
		addParam(createParamCentered<Rogan3PSWhite>(Vec(175.14, 113.55), module, Beads::PITCH_PARAM));

		addParam(createParamCentered<Rogan1PSWhite>(Vec(71.79, 179.2), module, Beads::SIZE_PARAM));
		addParam(createParamCentered<Rogan1PSWhite>(Vec(137.59, 179.2), module, Beads::SHAPE_PARAM));

		addParam(createParamCentered<Trimpot>(Vec(43.84+20.11/2, 238.345), module, Beads::TIME_ATTENUVERTER_PARAM));
		addParam(createParamCentered<Trimpot>(Vec(77.7+20.11/2, 238.345), module, Beads::SIZE_ATTENUVERTER_PARAM));
		addParam(createParamCentered<Trimpot>(Vec(111.56+20.11/2, 238.345), module, Beads::SHAPE_ATTENUVERTER_PARAM));
		addParam(createParamCentered<Trimpot>(Vec(145.42+20.11/2, 238.345), module, Beads::PITCH_ATTENUVERTER_PARAM));

		feedbackKnob = createParamCentered<Trimpot>(Vec(20.035, 179.205), module, Beads::FEEDBACK_PARAM);
		dryWetKnob = createParamCentered<Trimpot>(Vec(189.335, 179.205), module, Beads::DRY_WET_PARAM);
		reverbKnob = createParamCentered<Trimpot>(Vec(189.335, 227.115), module, Beads::REVERB_PARAM);
		addParam(feedbackKnob);
		addParam(dryWetKnob);
		addParam(reverbKnob);

		gainKnob = createParamCentered<TrimpotRed>(Vec(20.035, 179.205), module, Beads::IN_GAIN_PARAM);
		gainKnob->visible = false;
		addParam(gainKnob);
		amountKnobs[0] = createParamCentered<TrimpotGreen>(Vec(20.035, 179.205), module, Beads::FEEDBACK_AMOUNT_PARAM);
		amountKnobs[1] = createParamCentered<TrimpotGreen>(Vec(189.335, 179.205), module, Beads::DRY_WET_AMOUNT_PARAM);
		amountKnobs[2] = createParamCentered<TrimpotGreen>(Vec(189.335, 227.115), module, Beads::REVERB_AMOUNT_PARAM);
		for (int i = 0; i < 3; i++) {
			amountKnobs[i]->visible = false;
			addParam(amountKnobs[i]);
		}

		addInput(createInputCentered<PJ301MPort>(Vec(10.43+19.22/2, 285.37), module, Beads::ASSIGN_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(44.29+19.22/2, 285.37), module, Beads::TIME_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(78.15+19.22/2, 285.37), module, Beads::SIZE_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(112+19.22/2, 285.37), module, Beads::SHAPE_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(145.86+19.22/2, 285.37), module, Beads::PITCH_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(179.72+19.22/2, 285.37), module, Beads::DENSITY_INPUT));

		addInput(createInputCentered<PJ301MPort>(Vec(10.43+19.22/2, 328.54), module, Beads::IN_L_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(44.29+19.22/2, 328.54), module, Beads::IN_R_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(78.15+19.22/2, 328.54), module, Beads::FREEZE_INPUT));
		addInput(createInputCentered<PJ301MPort>(Vec(112+19.22/2, 328.54), module, Beads::SEED_INPUT));
		addOutput(createOutputCentered<PJ301MPort>(Vec(155.47, 328.54), module, Beads::OUT_L_OUTPUT));
		addOutput(createOutputCentered<PJ301MPort>(Vec(188.83, 328.54), module, Beads::OUT_R_OUTPUT));
	}

	void step() override {
		Beads *module = dynamic_cast<Beads*>(this->module);
		int view = module ? module->editView.load() : 0;
		if (currentView != view)
		{
			feedbackKnob->visible = view == 0;
			dryWetKnob->visible = view != 2;
			reverbKnob->visible = view != 2;
			gainKnob->visible = view == 1;
			for (int i = 0; i < 3; i++)
				amountKnobs[i]->visible = view == 2;
			currentView = view;
			panel->dirty = true;
		}
		ModuleWidget::step();
	}

	void appendContextMenu(Menu *menu) override {
		Beads *module = dynamic_cast<Beads*>(this->module);
		assert(module);

		menu->addChild(construct<MenuLabel>());
		menu->addChild(construct<MenuLabel>(&MenuLabel::text, "Media / Quality"));
		menu->addChild(construct<BeadsQualityItem>(&MenuItem::text, "Scorched cassette", &BeadsQualityItem::module, module, &BeadsQualityItem::media, 0));
		menu->addChild(construct<BeadsQualityItem>(&MenuItem::text, "Sunny tape", &BeadsQualityItem::module, module, &BeadsQualityItem::media, 1));
		menu->addChild(construct<BeadsQualityItem>(&MenuItem::text, "Cold digital", &BeadsQualityItem::module, module, &BeadsQualityItem::media, 2));
		menu->addChild(construct<BeadsQualityItem>(&MenuItem::text, "Bright digital", &BeadsQualityItem::module, module, &BeadsQualityItem::media, 3));

		menu->addChild(construct<MenuLabel>());
		menu->addChild(construct<BeadsToggleItem>(&MenuItem::text, "Seed latched", &BeadsToggleItem::module, module, &BeadsToggleItem::which, 1));
		menu->addChild(construct<BeadsToggleItem>(&MenuItem::text, "Gate output on the right channel", &BeadsToggleItem::module, module, &BeadsToggleItem::which, 0));

		menu->addChild(construct<MenuLabel>());
		menu->addChild(construct<MenuLabel>(&MenuLabel::text, "Edit Knobs"));
		menu->addChild(construct<BeadsViewItem>(&MenuItem::text, "Mix controls", &BeadsViewItem::module, module, &BeadsViewItem::view, 0));
		menu->addChild(construct<BeadsViewItem>(&MenuItem::text, "Input gain", &BeadsViewItem::module, module, &BeadsViewItem::view, 1));
		menu->addChild(construct<BeadsViewItem>(&MenuItem::text, "CV amounts", &BeadsViewItem::module, module, &BeadsViewItem::view, 2));

		menu->addChild(construct<MenuLabel>());
		menu->addChild(createIndexSubmenuItem("Manual Gain",
			{"Off", "Minimum", "25%", "50%", "75%", "Maximum"},
			[=]() { return gainMenuIndex(module); },
			[=](size_t index) { setGainMenuIndex(module, index); }));
	}
};


Model *modelBeads = createModel<Beads, BeadsWidget>("Beads");
