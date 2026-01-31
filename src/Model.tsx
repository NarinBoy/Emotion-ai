import * as tf from "@tensorflow/tfjs";
// import "@tensorflow/tfjs-backend-webgl";
import { useEffect, useState } from "react";


const testPredict = async (model: tf.LayersModel) => {
    // สร้างภาพ dummy ขาวดำ 48x48
    const dummyInput = tf.zeros([1, 48, 48, 1]);

    const output = model.predict(dummyInput) as tf.Tensor;

    const data = await output.data();
    console.log("🔮 prediction:", data);

    tf.dispose([dummyInput, output]);
};



const model = await tf.loadLayersModel(
    "/model/model.json",
    { strict: false }
);

await testPredict(model);



const Model = () => {
    const [status, setStatus] = useState("Loading model...");

    useEffect(() => {
        const loadModel = async () => {
            try {
                await tf.ready();

                // ⭐ เปลี่ยนตรงนี้
                await tf.setBackend("cpu");
                console.log("tf backend:", tf.getBackend());

                const model = await tf.loadLayersModel(
                    "/model/model.json",
                    { strict: false }
                );

                console.log("model loaded");
                setStatus("Model Ready ✅");
            } catch (err) {
                console.error(err);
                setStatus("Model Load Failed ❌");
            }
        };

        loadModel();
    }, []);

    return <div>{status}</div>;
};

export default Model;
