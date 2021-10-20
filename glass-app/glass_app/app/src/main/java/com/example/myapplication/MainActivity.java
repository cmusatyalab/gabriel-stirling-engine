package com.example.myapplication;


import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.view.PreviewView;

import com.google.protobuf.Any;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;



import java.util.Locale;
import java.util.function.Consumer;

import edu.cmu.cs.gabriel.camera.CameraCapture;
import edu.cmu.cs.gabriel.camera.ImageViewUpdater;
import edu.cmu.cs.gabriel.camera.YuvToJPEGConverter;
import edu.cmu.cs.gabriel.client.comm.ServerComm;
import edu.cmu.cs.gabriel.client.results.ErrorType;
import edu.cmu.cs.gabriel.protocol.Protos;
import edu.cmu.cs.sterling.Protos.Step;
import edu.cmu.cs.sterling.Protos.ToClientExtras;
import edu.cmu.cs.sterling.Protos.ToServerExtras;
import edu.cmu.cs.sterling.Protos.ViewfinderChange;
import edu.cmu.cs.sterling.Protos.ViewfinderStatus;

import com.example.glass.ui.GlassGestureDetector;
import com.example.glass.ui.GlassGestureDetector.Gesture;
import com.example.glass.ui.GlassGestureDetector.OnGestureListener;


public class MainActivity extends AppCompatActivity implements OnGestureListener{
    private static final String TAG = "MainActivity";
    private static final String SOURCE = "stirling_glass";
    private static final int PORT = 9099;

    private static final int WIDTH = 1920;
    private static final int HEIGHT = 1080;

    private Step step;
    private ViewfinderStatus viewfinderStatus = ViewfinderStatus.IsOn;
    private ViewfinderChange viewfinderChange;
    private static int framesSameHash = 0;
    private static int framesCompletedStep = 0;
    private static int detectedFrames = 0;
    private static int undetectedFrames = 0;
    private static boolean goBack = false;
    private static boolean lastFrameUndetected = false;
    private static String lastHash = "";


    private TextToSpeech textToSpeech;
    private ImageViewUpdater cropViewUpdater;
    private ImageViewUpdater instructionViewUpdater;

    private PreviewView viewFinder;
    private ImageView cropView;

    private ServerComm serverComm;
    private YuvToJPEGConverter yuvToJPEGConverter;
    private CameraCapture cameraCapture;

    private GlassGestureDetector glassGestureDetector;

    private final Consumer<Protos.ResultWrapper> consumer = resultWrapper -> {
        try {
            ToClientExtras toClientExtras = ToClientExtras.parseFrom(
                    resultWrapper.getExtras().getValue());
            step = toClientExtras.getStep();
            framesSameHash = toClientExtras.getFramesSameHash();
            framesCompletedStep = toClientExtras.getFramesCompletedStep();
            viewfinderChange = toClientExtras.getViewfinderChange();
            detectedFrames = toClientExtras.getDetectedFrames();
            undetectedFrames = toClientExtras.getUndetectedFrames();
            lastHash = toClientExtras.getLastHash();
            lastFrameUndetected = toClientExtras.getLastFrameUndetected();
            String speech = toClientExtras.getSpeech();
            if (!speech.isEmpty()) {
                this.textToSpeech.speak(speech, TextToSpeech.QUEUE_ADD, null, null);
                // this.sendingSwitch.post(() -> this.sendingSwitch.setChecked(false));
            }

            ByteString image = toClientExtras.getImage();
            if (!image.isEmpty()) {
                instructionViewUpdater.accept(image);
            }
            if (viewfinderChange == ViewfinderChange.TurnOn){
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        viewFinder.setVisibility(View.VISIBLE);
                        cropView.setVisibility(View.INVISIBLE);
                    }
                });
                viewfinderStatus = ViewfinderStatus.IsOn;
            }
            else if (viewfinderChange == ViewfinderChange.TurnOff){
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        viewFinder.setVisibility(View.INVISIBLE);
                        cropView.setVisibility(View.VISIBLE);
                    }
                });
                viewfinderStatus = ViewfinderStatus.IsOff;
            }

        } catch (InvalidProtocolBufferException e) {
            Log.e(TAG, "Protobuf parse error", e);
        }

        if (resultWrapper.getResultsCount() == 0) {
            return;
        }

        Protos.ResultWrapper.Result result = resultWrapper.getResults(0);
        ByteString jpegByteString = result.getPayload();
        cropViewUpdater.accept(jpegByteString);
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        viewFinder = findViewById(R.id.viewFinder);
        cropView = findViewById(R.id.cropView);
        cropViewUpdater = new ImageViewUpdater(cropView);
        ImageView instructionView = findViewById(R.id.instructionView);
        instructionViewUpdater = new ImageViewUpdater(instructionView);

        Consumer<ErrorType> onDisconnect = errorType -> {
            Log.e("MainActivity", "Disconnect Error: " + errorType.name());
            finish();
        };
        serverComm = ServerComm.createServerComm(
                consumer, BuildConfig.GABRIEL_HOST, PORT, getApplication(), onDisconnect);

        TextToSpeech.OnInitListener onInitListener = i -> {
            textToSpeech.setLanguage(Locale.US);

            ToServerExtras toServerExtras = ToServerExtras.newBuilder().setStep(Step.START).setViewfinderStatus(ViewfinderStatus.IsOn).setGoBack(false).build();
            Protos.InputFrame inputFrame = Protos.InputFrame.newBuilder()
                    .setExtras(pack(toServerExtras))
                    .build();

            // We need to wait for textToSpeech to be initialized before asking for the first
            // instruction.
            serverComm.send(inputFrame, SOURCE, /* wait */ true);
        };
        this.textToSpeech = new TextToSpeech(this, onInitListener);

        yuvToJPEGConverter = new YuvToJPEGConverter(this, 100);
        cameraCapture = new CameraCapture(this, analyzer, WIDTH, HEIGHT, viewFinder);

        glassGestureDetector = new GlassGestureDetector(this, this);
    }

    @Override
    public boolean onGesture(Gesture gesture) {
        switch (gesture) {
            case SWIPE_BACKWARD:
                serverComm.sendSupplier(() -> {
                    goBack = true;
                    ToServerExtras toServerExtras = ToServerExtras.newBuilder()
                            .setStep(step)
                            .setGoBack(goBack)
                            .setFramesCompletedStep(MainActivity.framesCompletedStep)
                            .setFramesSameHash(MainActivity.framesSameHash)
                            .setViewfinderStatus(MainActivity.this.viewfinderStatus)
                            .setDetectedFrames(MainActivity.detectedFrames)
                            .setUndetectedFrames(MainActivity.undetectedFrames)
                            .setLastHash(MainActivity.lastHash)
                            .build();
                    goBack = false;
                    return Protos.InputFrame.newBuilder()
                            .setPayloadType(Protos.PayloadType.IMAGE)
                            .setExtras(pack(toServerExtras))
                            .build();
                }, SOURCE, /* wait */ true);
                return true;

            case SWIPE_DOWN:
                finish();
                return true;
            default:
                return false;
        }
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent ev) {
        if (glassGestureDetector.onTouchEvent(ev)) {
            return true;
        }
        return super.dispatchTouchEvent(ev);
    }



    // Based on
    // https://github.com/protocolbuffers/protobuf/blob/master/src/google/protobuf/compiler/java/java_message.cc#L1387
    public static Any pack(ToServerExtras toServerExtras) {
        return Any.newBuilder()
                .setTypeUrl("type.googleapis.com/sterling.ToServerExtras")
                .setValue(toServerExtras.toByteString())
                .build();
    }

    final private ImageAnalysis.Analyzer analyzer = new ImageAnalysis.Analyzer() {
        @Override
        public void analyze(@NonNull ImageProxy image) {
//            if (!sendingSwitch.isChecked()) {
//                image.close();
//                return;
//            }
//            Log.i(TAG, "sending");

            serverComm.sendSupplier(() -> {
                ByteString jpegByteString = yuvToJPEGConverter.convert(image);

                ToServerExtras toServerExtras = ToServerExtras.newBuilder()
                        .setStep(MainActivity.this.step)
                        .setFramesCompletedStep(MainActivity.framesCompletedStep)
                        .setFramesSameHash(MainActivity.framesSameHash)
                        .setViewfinderStatus(MainActivity.this.viewfinderStatus)
                        .setDetectedFrames(MainActivity.detectedFrames)
                        .setUndetectedFrames(MainActivity.undetectedFrames)
                        .setLastHash(MainActivity.lastHash)
                        .setLastFrameUndetected(MainActivity.lastFrameUndetected)
                        .build();

                return Protos.InputFrame.newBuilder()
                        .setPayloadType(Protos.PayloadType.IMAGE)
                        .addPayloads(jpegByteString)
                        .setExtras(pack(toServerExtras))
                        .build();
            }, SOURCE, /* wait */ false);

            // The image has either been sent or skipped. It is therefore safe to close the image.
            image.close();
        }
    };

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraCapture.shutdown();
    }

}