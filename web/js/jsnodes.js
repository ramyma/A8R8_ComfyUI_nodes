import { app } from "../../../scripts/app.js";
import { applyTextReplacements } from "../../../scripts/utils.js";

app.registerExtension({
	name: "A8R8.jsnodes",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if(!nodeData?.category?.startsWith("A8R8")) {
			return;
		  }
		switch (nodeData.name) {

			case "AttentionCoupleRegions":

				nodeType.prototype.onNodeCreated = function () {
				this._type = "ATTENTION_COUPLE_REGION"
				this.inputs_offset = nodeData.name.includes("selective")?1:0
				this.addWidget("button", "Update inputs", null, () => {
					if (!this.inputs) {
						this.inputs = [];
					}
					const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
						if(target_number_of_inputs===this.inputs.length)return; // already set, do nothing

						if(target_number_of_inputs < this.inputs.length){
							for(let i = this.inputs.length; i>=this.inputs_offset+target_number_of_inputs; i--)
									this.removeInput(i)
						}
						else{
							for(let i = this.inputs.length+1-this.inputs_offset; i <= target_number_of_inputs; ++i)
								this.addInput(`region_${i}`, this._type)
						}
					});
				}
				break;
						
		}	
		
	}
});