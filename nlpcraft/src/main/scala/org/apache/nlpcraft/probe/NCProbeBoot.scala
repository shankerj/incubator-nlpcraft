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

package org.apache.nlpcraft.probe

import java.util.concurrent.CompletableFuture

import com.typesafe.config.{Config, ConfigFactory, ConfigValueFactory}
import com.typesafe.scalalogging.LazyLogging
import io.opencensus.trace.Span
import org.apache.nlpcraft.common.ascii.NCAsciiTable
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.nlp.core.NCNlpCoreManager
import org.apache.nlpcraft.common.nlp.dict.NCDictionaryManager
import org.apache.nlpcraft.common.nlp.numeric.NCNumericManager
import org.apache.nlpcraft.common.opencensus.NCOpenCensusTrace
import org.apache.nlpcraft.common.extcfg.NCExternalConfigManager
import org.apache.nlpcraft.common.version.NCVersion
import org.apache.nlpcraft.common.{NCE, NCException, NCService, U}
import org.apache.nlpcraft.model.NCModel
import org.apache.nlpcraft.probe.mgrs.cmd.NCCommandManager
import org.apache.nlpcraft.probe.mgrs.conn.NCConnectionManager
import org.apache.nlpcraft.probe.mgrs.conversation.NCConversationManager
import org.apache.nlpcraft.probe.mgrs.deploy.NCDeployManager
import org.apache.nlpcraft.probe.mgrs.dialogflow.NCDialogFlowManager
import org.apache.nlpcraft.probe.mgrs.lifecycle.NCLifecycleManager
import org.apache.nlpcraft.probe.mgrs.model.NCModelManager
import org.apache.nlpcraft.probe.mgrs.nlp.NCProbeEnrichmentManager
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.dictionary.NCDictionaryEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.limit.NCLimitEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.model.NCModelEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.relation.NCRelationEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.sort.NCSortEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.stopword.NCStopWordEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.enrichers.suspicious.NCSuspiciousNounsEnricher
import org.apache.nlpcraft.probe.mgrs.nlp.validate.NCValidateManager
import org.apache.nlpcraft.common.ansi.NCAnsi._
import org.apache.nlpcraft.common._

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.compat.Platform.currentTime
import scala.util.control.Exception.{catching, ignoring}

/**
  * Probe loader.
  */
private [probe] object NCProbeBoot extends LazyLogging with NCOpenCensusTrace {
    private final val executionStart = System.currentTimeMillis()

    private val startedMgrs = mutable.Buffer.empty[NCService]

    @volatile private var started = false
    @volatile private var shutdownHook: Thread = _
    @volatile private var probeThread: Thread = _
    
    // This container designed only for internal usage (transfer common data between methods).
    private case class ProbeConfig(
        var id: String,
        var token: String,
        var upLink: (String, Integer),
        var downLink: (String, Integer),
        var jarsFolder: Option[String],
        var models: String,
        var lifecycle: Seq[String]
    ) {
        lazy val upLinkString = s"${upLink._1}:${upLink._2}"
        lazy val downLinkString = s"${downLink._1}:${downLink._2}"
        lazy val modelsSeq: Seq[String] = models.split(",").map(_.trim)
    }
    
    private def mkDefault(): Config = {
        import ConfigValueFactory._
        
        val prefix = "nlpcraft.probe"
        
        ConfigFactory.empty().
            withValue(s"$prefix.id", fromAnyRef("dflt.probe")).
            // This is the default token as in default company.
            // Note that this token must match the probe token from the company this probe
            // associated with. If changed from default, this token must be kept secure.
            withValue(s"$prefix.token", fromAnyRef("3141592653589793")).
            withValue(s"$prefix.upLink", fromAnyRef("localhost:8201")).
            withValue(s"$prefix.downLink", fromAnyRef("localhost:8202")).
            withValue(s"$prefix.model", fromAnyRef(null)).
            withValue(s"$prefix.models", fromAnyRef("")).
            withValue(s"$prefix.lifecycle", fromIterable(Seq().asJava)).
            withValue(s"$prefix.resultMaxSizeBytes", fromAnyRef(1048576)).
            withValue("nlpcraft.nlpEngine", fromAnyRef("opennlp"))
        
        // Following properties are 'null' by default:
        // -------------------------------------------
        // nlpcraft.probe.modelFactory
        // nlpcraft.probe.modelFactory
        // nlpcraft.probe.jarsFolder
        // nlpcraft.probe.modelFactory
        // nlpcraft.probe.jaegerThriftUrl
        // nlpcraft.probe.prometheusLink
        // nlpcraft.probe.stackdriverGoogleProjectId
        // nlpcraft.probe.zipkinV2Url
    }
    
    /**
      *
      * @param args
      * @param overrideCfg
      * @return
      */
    @throws[NCE]
    private def initializeConfig(args: Array[String], overrideCfg: Option[Config]): ProbeConfig = {
        NCConfigurable.initialize(
            overrideCfg,
            args.find(_.startsWith("-config=")) match {
                case Some(s) ⇒
                    val fileName = s.substring("-config=".length)
    
                    val f = new java.io.File(fileName)
    
                    if (!(f.exists && f.canRead && f.isFile))
                        throw new NCE(s"Specified probe configuration file does not exist or cannot be read: $fileName")
    
                    Some(fileName)

                case None ⇒ Some("nlpcraft.conf")
            },
            Some(mkDefault()),
            (cfg: Config) ⇒ cfg.hasPath("nlpcraft.probe")
        )
        
        object Cfg extends NCConfigurable {
            private final val prefix = "nlpcraft.probe"
            
            val id: String = getString(s"$prefix.id")
            val token: String = getString(s"$prefix.token")
            val upLink: (String, Integer) = getHostPort(s"$prefix.upLink")
            val downLink: (String, Integer) = getHostPort(s"$prefix.downLink")
            val jarsFolder: Option[String] = getStringOpt(s"$prefix.jarsFolder")
            val models: String = getString(s"$prefix.models")
            val lifecycle: Seq[String] = getStringList(s"$prefix.lifecycle")
        }
        
        ProbeConfig(
            Cfg.id,
            Cfg.token,
            Cfg.upLink,
            Cfg.downLink,
            Cfg.jarsFolder,
            Cfg.models,
            Cfg.lifecycle
        )
    }
    
    /**
      *
      * @param cfg Probe configuration.
      * @param fut
      */
    private def start0(cfg: ProbeConfig, fut: CompletableFuture[Integer]): Unit = {
        probeThread = Thread.currentThread()
        
        asciiLogo()
        ackConfig(cfg)
        
        catching(classOf[Throwable]) either startManagers(cfg) match {
            case Left(e) ⇒ // Exception.
                U.prettyError(logger, "Failed to start probe:", e)

                stop0()

                fut.complete(1)
            
            case _ ⇒ // Managers started OK.
                shutdownHook = new Thread() {
                    override def run(): Unit = {
                        logger.info("Executing shutdown hook...")

                        stop0()
                    }
                }
                
                Runtime.getRuntime.addShutdownHook(shutdownHook)
                
                ackStart()

                started = true

                fut.complete(0)

                // Wait indefinitely.
                while (started)
                    try
                        Thread.currentThread().join()
                    catch {
                        case _: InterruptedException ⇒ ()
                    }
        }
    
        logger.trace("Probe thread stopped OK.")
    }
    
    /**
      * 
      */
    private def stop0(): Unit = {
        ignoring(classOf[Throwable]) {
            if (shutdownHook != null)
                Runtime.getRuntime.removeShutdownHook(shutdownHook)
            
            stopManagers()
        }
        
        started = false

        U.interruptThread(probeThread)

        logger.info("Probe shutdown OK.")
    }
    
    /**
      * 
      */
    private def checkStarted(): Unit = 
        if (started)
            throw new NCException(s"Probe has already been started (only one probe per JVM is allowed).")
    
    /**
      *
      * @param cfgFile
      * @param fut
      */
    private [probe] def start(cfgFile: String, fut: CompletableFuture[Integer]): Unit = {
        checkStarted()
        
        val cfg = initializeConfig(Array(s"-config=$cfgFile"), None)
        
        new Thread() {
            override def run(): Unit = start0(cfg, fut)
        }.start()
    }
    
    /**
      *
      * @param mdlClasses
      * @param fut
      */
    private [probe] def start(
        mdlClasses: Array[java.lang.Class[_ <: NCModel]],
        fut: CompletableFuture[Integer]): Unit = {
        checkStarted()
    
        import ConfigValueFactory._
    
        val cfg = initializeConfig(
            Array.empty,
            Some(
                ConfigFactory.empty().withValue(
                    "nlpcraft.probe.models",
                    fromAnyRef(mdlClasses.map(_.getName).mkString(","))
                )
            )
        )
        
        new Thread() {
            override def run(): Unit = start0(cfg, fut)
        }.start()
    }
    
    /**
      * 
      * @param probeId Probe ID.
      * @param tok
      * @param upLinkStr
      * @param dnLinkStr
      * @param mdlClasses
      * @param fut
      */
    private [probe] def start(
        probeId: String,
        tok: String,
        upLinkStr: String,
        dnLinkStr: String,
        mdlClasses: Array[java.lang.Class[_ <: NCModel]],
        fut: CompletableFuture[Integer]): Unit = {
        checkStarted()
    
        object Cfg extends NCConfigurable {
            private final val prefix = "nlpcraft.probe"
        
            val id: String = probeId
            val token: String = tok
            val upLink: (String, Integer) = getHostPort(upLinkStr)
            val dnLink: (String, Integer) = getHostPort(dnLinkStr)
            val jarsFolder: Option[String] = getStringOpt(s"$prefix.jarsFolder")
            val models: String = mdlClasses.map(_.getName).mkString(",")
            val lifecycle: Seq[String] = getStringList(s"$prefix.lifecycle")
        }
    
        new Thread() {
            override def run(): Unit =
                start0(ProbeConfig(
                    Cfg.id,
                    Cfg.token,
                    Cfg.upLink,
                    Cfg.dnLink,
                    Cfg.jarsFolder,
                    Cfg.models,
                    Cfg.lifecycle),
                    fut
                )
        }
        .start()
    }
    
    /**
      * 
      */
    private [probe] def stop(): Unit = 
        if (started)
            stop0()
    
    /**
      *
      * @param args
      * @param fut
      */
    private [probe] def start(args: Array[String], fut: CompletableFuture[Integer]): Unit =
        start0(initializeConfig(args, None), fut)
    
    /**
      * Prints ASCII-logo.
      */
    private def asciiLogo() {
        val ver = NCVersion.getCurrent

        println(
            U.NL +
            U.asciiLogo() +
            s"${U.NL}" +
            s"Embedded Data Probe${U.NL}" +
            s"Version: ${ansiBold(ver.version)}${U.NL}" +
            s"${NCVersion.copyright}${U.NL}"
        )
    }

    /**
      *
      */
    private def ackConfig(cfg: ProbeConfig): Unit = {
        val tbl = NCAsciiTable()
        
        val ver = NCVersion.getCurrent

        tbl += (s"${ansiBlueFg}Probe ID$ansiReset", s"$ansiBold${cfg.id}$ansiReset")
        tbl += (s"${ansiBlueFg}Probe Token$ansiReset", cfg.token)
        tbl += (s"${ansiBlueFg}API Version$ansiReset", ver.version + ", " + ver.date.toString)
        tbl += (s"${ansiBlueFg}Down-Link$ansiReset", cfg.downLinkString)
        tbl += (s"${ansiBlueFg}Up-Link$ansiReset", cfg.upLinkString)
        tbl += (s"${ansiBlueFg}Lifecycle$ansiReset", cfg.lifecycle)
        tbl += (s"${ansiBlueFg}Models (${cfg.modelsSeq.size})$ansiReset" , cfg.modelsSeq)
        tbl += (s"${ansiBlueFg}JARs Folder$ansiReset", cfg.jarsFolder.getOrElse(""))

        tbl.info(logger, Some("Probe Configuration:"))
    }
    
    /**
      * Asks server start.
      */
    private def ackStart() {
        val dur = s"[${U.format((currentTime - executionStart) / 1000.0, 2)} sec]"
        
        val tbl = NCAsciiTable()
        
        tbl.margin(bottom = 1, top = 1)
        
        tbl += s"Probe started ${b(dur)}"
        
        tbl.info(logger)
    }

    /**
      *
      * @return
      */
    private def startManagers(cfg: ProbeConfig): Unit = {
        // Lifecycle callback outside of tracing span.
        NCLifecycleManager.start()
        NCLifecycleManager.onInit()
        
        startScopedSpan("startManagers") { span ⇒
            val ver = NCVersion.getCurrent
            
            addTags(
                span,
                "id" → cfg.id,
                "token" → cfg.token,
                "uplink" → cfg.upLinkString,
                "downlink" → cfg.downLinkString,
                "relVer" → ver.version,
                "relDate" → ver.date.toString,
                "models" → cfg.models,
                "lifecycle" → cfg.lifecycle.mkString(","),
                "jarFolder" → cfg.jarsFolder
            )

            startedMgrs += NCExternalConfigManager.start(span)
            startedMgrs += NCNlpCoreManager.start(span)
            startedMgrs += NCNumericManager.start(span)
            startedMgrs += NCDeployManager.start(span)
            startedMgrs += NCModelManager.start(span)
            startedMgrs += NCCommandManager.start(span)
            startedMgrs += NCDictionaryManager.start(span)
            startedMgrs += NCStopWordEnricher.start(span)
            startedMgrs += NCModelEnricher.start(span)
            startedMgrs += NCLimitEnricher.start(span)
            startedMgrs += NCSortEnricher.start(span)
            startedMgrs += NCRelationEnricher.start(span)
            startedMgrs += NCSuspiciousNounsEnricher.start(span)
            startedMgrs += NCValidateManager.start(span)
            startedMgrs += NCDictionaryEnricher.start(span)
            startedMgrs += NCConversationManager.start(span)
            startedMgrs += NCProbeEnrichmentManager.start(span)
            startedMgrs += NCConnectionManager.start(span)
            startedMgrs += NCDialogFlowManager.start(span)
        }
    }

    private def stopManager(mgr: NCService, span: Span): Unit =
        try
            mgr.stop(span)
        catch {
            case e: Throwable ⇒ U.prettyError(logger, s"Failed to stop manager: ${mgr.name}", e)
        }

    /**
      *
      */
    private def stopManagers(): Unit = {
        startScopedSpan("stopManagers") { span ⇒
            startedMgrs.synchronized {
                try
                    startedMgrs.reverseIterator.foreach(stopManager(_, span))
                finally
                    startedMgrs.clear()
            }

            // Lifecycle callback outside of tracing span.
            NCLifecycleManager.onDiscard()
            stopManager(NCLifecycleManager, span)
        }
    }
}
