/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.nlpcraft.server.ml

import io.opencensus.trace.Span
import org.apache.nlpcraft.common.{NCE, NCService}
import org.apache.nlpcraft.server.mdo.NCModelMlConfigMdo
import org.apache.nlpcraft.server.opencensus.NCOpenCensusServerStats

/**
  * TODO:
  */
object NCMlManager extends NCService with NCOpenCensusServerStats {
    override def start(parent: Span = null): NCService = startScopedSpan("start", parent) { _ ⇒
        super.start()
    }

    override def stop(parent: Span = null): Unit = startScopedSpan("stop", parent) { _ ⇒
        super.stop()
    }

    @throws[NCE]
    def makeModelConfig(elems: Map[String, Set[String]], examples: Set[String]): NCModelMlConfigMdo = {
        // TODO:
        NCModelMlConfigMdo(elems, examples)
    }
}
