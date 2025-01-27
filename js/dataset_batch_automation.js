import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "DatasetBatchAutomation",
    setup() {
        console.log("DatasetBatchAutomation: setup() called - Extension is loading");

        api.addEventListener("dataset_row_processed", (event) => {
            console.log("DatasetBatchAutomation: dataset_row_processed event received.", event.detail);

            // Find the DatasetBatchNode by its type name
            const datasetBatchNode = Object.values(app.graph._nodes).find(
                (node) => node.type === "DatasetBatchNode"
            );

            if (datasetBatchNode) {
                console.log("DatasetBatchAutomation: DatasetBatchNode found in workflow.");

                // Queue the next prompt automatically
                app.queuePrompt(0, 1);
                console.log("DatasetBatchAutomation: Queued next prompt.");
            } else {
                console.warn(
                    `DatasetBatchAutomation: DatasetBatchNode not found in workflow.`
                );
            }
        });
    },
});